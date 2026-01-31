#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod
import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower, build_text_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from transformers import CLIPTextModel

# paper-style anchors (your updated version)
from llava.model.task_anchor import DistributedGraphEvolvingAnchors


def _is_rank0():
    if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


class LlavaMetaModel:
    """
    Single-file mitigation for catastrophic forgetting under:
    - separate script per task
    - one expert per task

    What we do here (without touching training script):
    1) Auto load/save anchor_manager state across runs (if state_dict/load_state_dict exist).
    2) During training, force expert_weight to one-hot on current task so only current expert is used
       (helps prevent old expert degradation if your MoE-LoRA uses expert_weight in forward).
    3) Slow down anchor EMA + reduce refine frequency with warmup to avoid anchor drift.
    """

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)

        # persistent anchors + counters
        self.anchor_manager = None
        self.anchor_step = 0
        self._anchors_loaded = False

        # cache CLIPTextModel for eval
        self.eval_clip_text_tower = None

        # ---- knobs (env overridable) ----
        # where to save anchors
        self.anchor_dir = os.environ.get("LLaVA_ANCHOR_DIR", "./checkpoints/anchors")
        # how often to save anchors during training (rank0 only)
        self.anchor_save_every = int(os.environ.get("LLaVA_ANCHOR_SAVE_EVERY", "200"))
        # how often to refine anchors (after warmup)
        self.anchor_refine_every = int(os.environ.get("LLaVA_ANCHOR_REFINE_EVERY", "500"))
        self.anchor_warmup = int(os.environ.get("LLaVA_ANCHOR_WARMUP", "1000"))
        # slower EMA to reduce drift / forgetting
        self.anchor_momentum = float(os.environ.get("LLaVA_ANCHOR_MOMENTUM", "0.01")) #1-0.01

        os.makedirs(self.anchor_dir, exist_ok=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_text_tower(self):
        text_tower = getattr(self, "text_tower", None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def get_eval_clip_text_tower(self, ckpt_path: str, device: torch.device):
        if self.eval_clip_text_tower is None:
            self.eval_clip_text_tower = CLIPTextModel.from_pretrained(ckpt_path).to(device)
            self.eval_clip_text_tower.requires_grad_(False)
            self.eval_clip_text_tower.eval()
        return self.eval_clip_text_tower

    # ----------------- Anchor persistence helpers -----------------
    def _anchor_ckpt_path_for_task(self, task_id: int):
        # save one file per task stage; later tasks can load the latest <= current task_id
        return os.path.join(self.anchor_dir, f"anchors_task{int(task_id)}.pt")

    def _find_latest_anchor_ckpt(self, task_id: int):
        # pick the largest t <= task_id existing
        for t in range(int(task_id), -1, -1):
            p = self._anchor_ckpt_path_for_task(t)
            if os.path.exists(p):
                return p
        return None

    def _maybe_load_anchors(self, cur_task: int):
        """
        Auto-load anchors once per run.
        - loads the latest anchors_task{<=cur_task}.pt if exists.
        """
        if self._anchors_loaded:
            return
        self._anchors_loaded = True

        if not hasattr(self.anchor_manager, "load_state_dict"):
            return

        ckpt = self._find_latest_anchor_ckpt(cur_task)
        if ckpt is None:
            return

        try:
            sd = torch.load(ckpt, map_location="cpu")
            self.anchor_manager.load_state_dict(sd, strict=False)
            if _is_rank0():
                print(f"[Anchor] auto-loaded from {ckpt}")
        except Exception as e:
            if _is_rank0():
                print(f"[Anchor] auto-load failed from {ckpt}: {e}")

    def _maybe_save_anchors(self, cur_task: int):
        """
        Auto-save anchors periodically (rank0 only).
        """
        if not _is_rank0():
            return
        if not hasattr(self.anchor_manager, "state_dict"):
            return
        # save to current task file (overwrite)
        ckpt = self._anchor_ckpt_path_for_task(cur_task)
        try:
            sd = self.anchor_manager.state_dict()
            torch.save(sd, ckpt)
        except Exception as e:
            print(f"[Anchor] auto-save failed to {ckpt}: {e}")

    def get_distribute_anchors(self):
        # IMPORTANT: do NOT recreate every call
        if self.anchor_manager is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.anchor_manager = DistributedGraphEvolvingAnchors(
                task_ids=[0, 1, 2, 3, 4, 5],
                K_per_task=5, 
                feat_dim=1536,         # fused = [img(768), txt(768)] -> 1536
                device=device,
                topk_graph=21,         
            ).to(device)
        return self.anchor_manager

    # ----------------- init modules -----------------
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"), strict=False)

    def initialize_text_modules(self, model_args, fsdp=None):
        text_tower = model_args.text_tower

        if self.get_text_tower() is None:
            text_tower = build_text_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.text_tower = [text_tower]
            else:
                self.text_tower = text_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                text_tower = self.text_tower[0]
            else:
                text_tower = self.text_tower
            text_tower.load_model()


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def get_distribute_anchors(self):
        return self.get_model().get_distribute_anchors()

    def encode_images(self, images):
        clip_image_features, image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return clip_image_features.to(self.device), image_features.to(self.device)

    def _set_expert_weight(self, weight_list):
        """
        Set expert_weight to last layer projections.
        If your CKM MoE-LoRA consumes expert_weight, this controls which expert is active.
        """
        proj_names = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        for proj_name in proj_names:
            if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj_layer = getattr(self.model.layers[-1].self_attn, proj_name)
            else:
                proj_layer = getattr(self.model.layers[-1].mlp, proj_name)
            proj_layer.expert_weight = weight_list

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_guide_features, image_features = self.encode_images(images)

        assert image_features.shape[1] == 576, "vision tower not a with projection version."

        # ---------- Text tower for CLIP text features ----------
        if self.training:
            text_tower = self.get_text_tower()
        else:
            text_tower = self.get_model().get_eval_clip_text_tower(
                ckpt_path="/root/Desktop/code/CKM/CKM-LLaVA-main/clip-vit-large-patch14-336",
                device=self.device,
            )

        # ---------- Prepare CLIP text inputs ----------
        input_pad = np.where(
            input_ids.cpu().detach().numpy() != -200,
            input_ids.cpu().detach().numpy(),
            self.tokenizer.pad_token_id,
        )
        decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden_inputs = ["\n".join(decode_input.split("\n")[1:]) for decode_input in decoded_inputs]
        decoded_clip_inputs = [decode_input.split(" ASSISTANT")[0] for decode_input in decoded_hidden_inputs]

        clip_text_inputs = self.clip_tokenizer(
            decoded_clip_inputs,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        # text_guide_features: [bs, 768]
        if self.training:
            text_guide_features = text_tower(clip_text_inputs)
        else:
            text_guide_features = text_tower(**(clip_text_inputs.to(self.device))).pooler_output.to(self.device)

        img_feat = F.normalize(image_guide_features, dim=-1)
        txt_feat = F.normalize(text_guide_features, dim=-1)
        fused_features = torch.cat([img_feat, txt_feat], dim=-1)  # [B, 1536]

        # ---------- Anchors (persistent) ----------
        anchor_manager = self.get_distribute_anchors()

        # current task id (your runner must set self.cur_task)
        task_id = int(self.cur_task)

        # ✅ auto-load anchors once per run (cross-script persistence)
        self.get_model()._maybe_load_anchors(task_id)

        if self.training:
            # (1) Force one-hot routing to current expert during training
            #     This is the biggest "single-file" mitigation for forgetting when each task has its own expert.
            T = len(anchor_manager.task_ids)
            one_hot = [0.0] * T
            if 0 <= task_id < T:
                one_hot[task_id] = 1.0
            self._set_expert_weight(one_hot)

            # (2) Update anchors with slower EMA (reduce drift)
            task_labels = [task_id] * fused_features.size(0)
            # rank0 updates anchors to avoid divergence across ranks
            if _is_rank0():
                anchor_manager.online_update_batch(fused_features, task_labels, momentum=self.get_model().anchor_momentum)

                # (3) Refine anchors less frequently and only after warmup
                self.get_model().anchor_step += 1
                if (self.get_model().anchor_step > self.get_model().anchor_warmup and
                        (self.get_model().anchor_step % self.get_model().anchor_refine_every) == 0):
                    anchor_manager.refine_across_tasks()

                # (4) Periodic auto-save anchors (cross-script persistence)
                if (self.get_model().anchor_step % self.get_model().anchor_save_every) == 0:
                    self.get_model()._maybe_save_anchors(task_id)
            else:
                # still advance step counter
                self.get_model().anchor_step += 1

        else:
            # ✅ Paper Eq(16): alpha_n from anchors
            alpha_dict = anchor_manager.compute_expert_weights(fused_features[0])  # {tid: tensor}
            task_ids = anchor_manager.task_ids
            compute_expert_weight = [float(alpha_dict[tid].item()) for tid in task_ids]

            # ensure sum=1
            s = sum(compute_expert_weight) + 1e-12
            compute_expert_weight = [w / s for w in compute_expert_weight]

            # (optional) mild calibration to avoid over-confident routing (helps stability)
            # T_calib > 1 flattens
            T_calib = float(os.environ.get("LLaVA_ALPHA_T", "1.0"))
            if T_calib != 1.0:
                arr = np.array(compute_expert_weight, dtype=np.float64)
                arr = np.power(arr + 1e-12, 1.0 / T_calib)
                arr = arr / (arr.sum() + 1e-12)
                compute_expert_weight = arr.tolist()

            self._set_expert_weight(compute_expert_weight)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # ---------- Below is your original sequence packing logic ----------
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels_out = None
        else:
            new_labels_out = new_labels_padded

        if _attention_mask is None:
            attention_mask_out = None
        else:
            attention_mask_out = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids_out = None
        else:
            position_ids_out = position_ids

        return None, position_ids_out, attention_mask_out, past_key_values, new_input_embeds, new_labels_out

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Num new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
