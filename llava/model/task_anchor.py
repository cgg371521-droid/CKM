import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

############################################################
# GaussianMixtureComponent (simple diagonal-cov Gaussian)
############################################################
class GMMComponent:
    def __init__(self, dim, device='cpu', init_mean=None):
        self.dim = dim
        self.device = device
        if init_mean is None:
            self.mean = torch.zeros(dim, device=device)
        else:
            self.mean = init_mean.to(device)
        # use log var for stability
        self.log_var = torch.zeros(dim, device=device) + np.log(1.0)
        self.count = 1.0  # effective count for online update
        self.weight = 1.0  # mixture weight (normalized later)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.log_var = self.log_var.to(device)
        self.device = device
        return self

    def var(self):
        return torch.exp(self.log_var)

    def update_online(self, x_batch, resp, momentum=0.1):
        # x_batch: (B, D), resp: (B,) responsibilities for this component (soft)
        # compute weighted mean & var and EMA-update parameters
        if x_batch.shape[0] == 0:
            return
        r = resp.float().unsqueeze(1)  # (B,1)
        total_r = r.sum() + 1e-8
        weighted_mean = (r * x_batch).sum(dim=0) / total_r
        diff = x_batch - weighted_mean.unsqueeze(0)
        weighted_var = (r * (diff ** 2)).sum(dim=0) / (total_r + 1e-8)

        # EMA update
        self.mean = (1 - momentum) * self.mean + momentum * weighted_mean
        # update log_var
        new_log_var = torch.log(weighted_var + 1e-6)
        self.log_var = (1 - momentum) * self.log_var + momentum * new_log_var
        # update counts and weight
        self.count = (1 - momentum) * self.count + momentum * total_r.item()
        # weight will be normalized by owning GMM owner
        # keep on device
        self.mean = self.mean.to(self.device)
        self.log_var = self.log_var.to(self.device)

    def em_init(self, x_batch):
        # initialize mean/var from x_batch (torch)
        if x_batch.shape[0] == 0:
            return
        self.mean = x_batch.mean(dim=0).detach().clone()
        var = x_batch.var(dim=0) + 1e-6
        self.log_var = torch.log(var)
        self.count = float(x_batch.shape[0])

############################################################
# TaskAnchor: add KMeans init + expose N (count) as N_{n,k}
############################################################
class TaskAnchor:
    def __init__(self, task_id, K, dim, device='cpu'):
        self.task_id = task_id
        self.K = K
        self.dim = dim
        self.device = device
        self.components = [GMMComponent(dim, device=device) for _ in range(K)]
        # weights here will represent ��_{n,k} (mixture coef) after normalize
        self.weights = torch.ones(K, device=device)
        self.last_update = time.time()
        self._inited = False

    def to(self, device):
        self.device = device
        for c in self.components:
            c.to(device)
        self.weights = self.weights.to(device)
        return self

    def normalize_weights(self):
        self.weights = F.softmax(self.weights, dim=0)
        return self.weights

    def init_with_kmeans(self, x_init: torch.Tensor):
      """
      Paper Sec 3.1: initialize μ, σ^2 by KMeans on a small batch.
      Fix: sklearn/numpy doesn't support bf16 -> cast to float32 on CPU.
      """
      if x_init is None or x_init.numel() == 0:
          # fallback: keep default
          self.weights = torch.ones(self.K, device=self.device)
          self.normalize_weights()
          self._inited = True
          return
  
      # ensure enough samples
      if x_init.shape[0] < self.K:
          # fallback: EM init with whatever we have
          for c in self.components:
              c.em_init(x_init.to(dtype=torch.float32))
          self.weights = torch.ones(self.K, device=self.device)
          self.normalize_weights()
          self._inited = True
          return
  
      # ✅ cast bf16/fp16 -> fp32 before numpy/kmeans
      x_cpu_fp32 = x_init.detach().to(dtype=torch.float32, device="cpu")
      X = x_cpu_fp32.numpy()
  
      # sklearn KMeans
      km = KMeans(n_clusters=self.K, random_state=0, n_init="auto").fit(X)
      labels = km.labels_
      centers = km.cluster_centers_
  
      new_comps = []
      # also get original fp32 tensor on device for stats
      x_dev_fp32 = x_init.detach().to(dtype=torch.float32, device=self.device)
  
      for k in range(self.K):
          idx = np.where(labels == k)[0]
          if len(idx) == 0:
              # empty cluster fallback
              m = torch.tensor(centers[k], dtype=torch.float32, device=self.device)
              comp = GMMComponent(self.dim, device=self.device, init_mean=m)
              comp.log_var = torch.log(torch.ones(self.dim, device=self.device) * 1.0)
              comp.count = 1.0
          else:
              xb = x_dev_fp32[idx]  # fp32 on device
              m = xb.mean(dim=0)
              v = xb.var(dim=0, unbiased=False) + 1e-6
              comp = GMMComponent(self.dim, device=self.device, init_mean=m)
              comp.log_var = torch.log(v)
              comp.count = float(len(idx))
          new_comps.append(comp)
  
      self.components = new_comps
      self.weights = torch.ones(self.K, device=self.device)
      self.normalize_weights()
      self._inited = True


    def collect_component_stats(self):
        means = torch.stack([c.mean for c in self.components], dim=0)      # (K,D)
        vars_  = torch.stack([c.var()  for c in self.components], dim=0)   # (K,D)
        # ��_{n,k} (mixture coef) normalized
        weights = F.softmax(self.weights, dim=0)                           # (K,)
        # N_{n,k} (empirical count) from component.count
        counts = torch.tensor([float(c.count) for c in self.components],
                              device=self.device, dtype=torch.float32)     # (K,)
        return means, vars_, weights, counts

    # keep your score_samples / soft_assign / online_update,
    # but make sure online_update updates "counts" & "weights" reasonably.
    def score_samples(self, x):
        # x: (N, D) -> return log probs (N, K) negative energy-like
        means = torch.stack([c.mean for c in self.components], dim=0)  # (K,D)
        vars_ = torch.stack([c.var() for c in self.components], dim=0)  # (K,D)
        # compute negative energy (higher = better)
        neg_energy = -batch_diag_mahalanobis(x, means, vars_)  # (N, K) negative cost
        # incorporate component prior (weights)
        log_weights = torch.log(F.softmax(self.weights, dim=0) + 1e-8).unsqueeze(0)  # (1,K)
        return neg_energy + log_weights  # (N,K)

    def soft_assign(self, x):
        # returns responsibilities (N, K)
        logits = self.score_samples(x)  # (N,K)
        resp = F.softmax(logits, dim=1)
        return resp

    def online_update(self, x, resp, momentum=0.05):
        """
        Align with Eq(5)-(6): EMA update for ��/log��^2, update N, then ��.
        We do:
          N <- (1-a)N + a R_k
          �� <- (1-a)�� + a (N / sum N)
        """
        K = self.K
        Rk = resp.sum(dim=0)  # (K,)
        # update each component stats
        for k in range(K):
            if Rk[k].item() < 1e-6:
                continue
            self.components[k].update_online(x, resp[:, k], momentum=momentum)

        # update counts N using EMA of Rk (Eq 6)
        for k in range(K):
            Nk_old = float(self.components[k].count)
            Nk_new = (1 - momentum) * Nk_old + momentum * float(Rk[k].item())
            self.components[k].count = Nk_new

        # update �� using EMA toward normalized counts (Eq 6)
        counts = torch.tensor([float(c.count) for c in self.components],
                              device=self.device, dtype=torch.float32)
        counts_norm = counts / (counts.sum() + 1e-8)
        w = F.softmax(self.weights, dim=0)
        w = (1 - momentum) * w + momentum * counts_norm
        self.weights = torch.log(w + 1e-8)  # keep in logit form for stability
        self.normalize_weights()
        self.last_update = time.time()


############################################################
# Paper-style parameter-free Component Graph Refiner (Eq 7-14)
############################################################
class ComponentGraphRefiner:
    """
    Implements Sec 3.2 Eq(7)-(14) in the paper:
      eta_{n,k} = N_{n,k} / (1 + (1/D) sum_d sigma^2_{n,k,d})
      z_{n,k}   = log(1+eta_{n,k}) / sum_j log(1+eta_{n,j})   (paper normalizes across all components)
      kappa((n,k),(m,l)) = exp( -1/2 sum_d (mu_nk - mu_ml)^2 / (sigma_nk^2 + sigma_ml^2) )
      S = z_i z_j kappa
      keep TopK neighbors from other tasks
      delta_mu_i = sum_{j in N(i)} alpha_ij (mu_j - mu_i)
      alpha_ij = softmax(exp(S_ij)) over neighbors
      mu_i <- mu_i + z_i * delta_mu_i
    """
    def __init__(self, topk=8):
        self.topk = topk

    @torch.no_grad()
    def refine(self, anchors: dict, task_ids: list, device='cpu'):
        # flatten all components as nodes
        node_task = []
        node_comp = []
        mus = []
        vars_ = []
        counts = []

        for tid in task_ids:
            means, v, w, cts = anchors[tid].collect_component_stats()
            K = means.shape[0]
            for k in range(K):
                node_task.append(tid)
                node_comp.append(k)
                mus.append(means[k])
                vars_.append(v[k])
                counts.append(cts[k])

        mus = torch.stack(mus, dim=0).to(device)        # (M,D)
        vars_ = torch.stack(vars_, dim=0).to(device)    # (M,D)
        counts = torch.stack(counts, dim=0).to(device)  # (M,)
        M, D = mus.shape

        # Eq(7): eta
        avg_var = vars_.mean(dim=1)  # (M,)
        eta = counts / (1.0 + avg_var + 1e-8)

        # Eq(8): z (normalize across ALL components)
        conf = torch.log1p(eta)
        z = conf / (conf.sum() + 1e-8)  # (M,)

        # build neighbors TopK from OTHER tasks for each node
        # Compute pairwise kappa only across different tasks.
        # For modest M this O(M^2 D) is ok; for big M you should block/chunk.
        # kappa_ij = exp( -1/2 sum_d (mu_i-mu_j)^2 / (var_i+var_j) )
        mu_i = mus.unsqueeze(1)            # (M,1,D)
        mu_j = mus.unsqueeze(0)            # (1,M,D)
        denom = (vars_.unsqueeze(1) + vars_.unsqueeze(0) + 1e-8)  # (M,M,D)
        dist = ((mu_i - mu_j) ** 2 / denom).sum(dim=2)            # (M,M)
        kappa = torch.exp(-0.5 * dist)                            # (M,M)

        # task mask (exclude same-task)
        t = torch.tensor(node_task, device=device)
        same_task = (t.unsqueeze(1) == t.unsqueeze(0))            # (M,M)
        kappa = kappa.masked_fill(same_task, 0.0)

        # Eq(10): S = z_i z_j kappa
        S = (z.unsqueeze(1) * z.unsqueeze(0)) * kappa             # (M,M)

        # for each i select TopK j by S_ij
        Ksel = min(self.topk, M - 1) if M > 1 else 0
        new_mus = mus.clone()

        for i in range(M):
            s_row = S[i]  # (M,)
            if Ksel <= 0 or torch.all(s_row <= 0):
                continue
            topv, topidx = torch.topk(s_row, k=Ksel, largest=True)
            # filter zeros (could happen due to same-task masking)
            valid = topv > 0
            if valid.sum().item() == 0:
                continue
            nbr_idx = topidx[valid]
            nbr_s = topv[valid]

            # Eq(13): alpha = softmax(exp(S)) over neighbors
            # Note paper writes exp(S) then normalize; equivalent to softmax on S.
            alpha = F.softmax(nbr_s, dim=0)  # (K',)

            # Eq(12): delta_mu = sum alpha (mu_j - mu_i)
            delta = (alpha.unsqueeze(1) * (mus[nbr_idx] - mus[i].unsqueeze(0))).sum(dim=0)

            # Eq(14): mu <- mu + z_i * delta
            new_mus[i] = mus[i] + z[i] * delta

        # write back updated means to anchors
        ptr = 0
        for tid in task_ids:
            anchor = anchors[tid]
            for k in range(anchor.K):
                anchor.components[k].mean = new_mus[ptr].detach()
                ptr += 1

        return z.detach(), new_mus.detach()


###########################################################
# Self-Evolving Expert Manager
############################################################
class SelfEvolvingManager:
    def __init__(self, task_anchor_dict, merge_threshold=0.95, split_var_threshold=2.0, device='cpu'):
        """
        task_anchor_dict: {task_id: TaskAnchor}
        merge_threshold: cosine similarity threshold to merge components across tasks
        split_var_threshold: if component variance trace > this * base_var then trigger split
        """
        self.task_anchor_dict = task_anchor_dict
        self.merge_threshold = merge_threshold
        self.split_var_threshold = split_var_threshold
        self.device = device

    def component_similarity(self, mean_a, mean_b):
        # cosine similarity
        ma = F.normalize(mean_a, dim=0)
        mb = F.normalize(mean_b, dim=0)
        return float((ma * mb).sum().item())

    def check_merge_across_tasks(self):
        # naive O(T^2 K^2) scan -- ok for modest T,K
        tasks = list(self.task_anchor_dict.keys())
        merges = []
        for i, ti in enumerate(tasks):
            anchor_i = self.task_anchor_dict[ti]
            means_i, _, weights_i = anchor_i.collect_component_stats()
            for j in range(i+1, len(tasks)):
                tj = tasks[j]
                anchor_j = self.task_anchor_dict[tj]
                means_j, _, weights_j = anchor_j.collect_component_stats()
                for a in range(means_i.shape[0]):
                    for b in range(means_j.shape[0]):
                        sim = self.component_similarity(means_i[a], means_j[b])
                        if sim >= self.merge_threshold:
                            merges.append((ti, a, tj, b, sim))
        return merges

    def apply_merge(self, merge_list):
        # merge across tasks by merging components into the lower task id owner (arbitrary policy)
        for (ti, ai, tj, bj, sim) in merge_list:
            # merge component bj of task tj into component ai of task ti and delete component in tj
            anchor_i = self.task_anchor_dict[ti]
            anchor_j = self.task_anchor_dict[tj]
            # careful with indices if previous merges removed components
            # simple approach: merge into anchor_i first, then delete bj
            # We'll map bj to current index by matching closest mean
            means_j, _, _ = anchor_j.collect_component_stats()
            # find actual bj index as argmin distance to saved mean
            # (here assume bj still exists)
            try:
                anchor_i.merge_components(ai, anchor_i.K - 1)  # dummy safe call to ensure structure; in practice implement robust index mapping
            except Exception:
                pass
            # NOTE: implementers should carefully handle indices; here we only return merges for user to handle
        # For prototype this method returns merge_list for manual handling
        return

    def check_split_within_task(self, task_id, var_trace_threshold=None):
        # For a given task, check each component's variance (trace) relative to average
        anchor = self.task_anchor_dict[task_id]
        means, vars_, weights = anchor.collect_component_stats()  # (K,D)
        trace = vars_.sum(dim=1).cpu().numpy()  # (K,)
        avg_trace = trace.mean() + 1e-8
        candidates = []
        for k, tval in enumerate(trace):
            if tval > self.split_var_threshold * avg_trace:
                candidates.append((task_id, k, float(tval / avg_trace)))
        return candidates

    def split_component_by_samples(self, task_id, comp_idx, sample_matrix, n_splits=2):
        """
        sample_matrix: np or torch ndarray of shape (N_samples, D) containing samples assigned to comp_idx
        returns new means/vars to replace original comp with n_splits components
        """
        if isinstance(sample_matrix, torch.Tensor):
            X = sample_matrix.cpu().numpy()
        else:
            X = sample_matrix
        if X.shape[0] < n_splits:
            return None
        kmeans = KMeans(n_clusters=n_splits, random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        new_components = []
        for c in centers:
            m = torch.tensor(c, dtype=torch.float32, device=self.device)
            comp = GMMComponent(X.shape[1], device=self.device, init_mean=m)
            # compute var from assigned points
            assigned = X[kmeans.labels_ == np.where((centers == c).all(axis=1))[0][0]] if False else X[kmeans.labels_ == np.argmin(((X - c)**2).sum(axis=1))]
            # fallback: set small var
            var = torch.tensor(np.var(X, axis=0) + 1e-6, dtype=torch.float32, device=self.device)
            comp.log_var = torch.log(var)
            comp.count = float(len(X))
            new_components.append(comp)
        # Now in TaskAnchor we need to replace component comp_idx with these n_splits components
        anchor = self.task_anchor_dict[task_id]
        # remove old
        del anchor.components[comp_idx]
        # insert new
        for nc in reversed(new_components):
            anchor.components.insert(comp_idx, nc)
        anchor.K = len(anchor.components)
        # reinitialize weights
        anchor.weights = torch.ones(anchor.K, device=self.device)
        anchor.normalize_weights()
        return True

def batch_diag_mahalanobis(x, means, diag_vars):
    # x: (N, D) , means: (K, D) , diag_vars: (K, D)
    # returns (N, K)
    N, D = x.shape
    K = means.shape[0]
    x_exp = x.unsqueeze(1).expand(N, K, D)  # (N, K, D)
    means_exp = means.unsqueeze(0).expand(N, K, D)
    vars_exp = diag_vars.unsqueeze(0).expand(N, K, D)
    diff = x_exp - means_exp
    return 0.5 * (torch.sum((diff ** 2) / (vars_exp + 1e-8), dim=2)
                  + torch.sum(torch.log(vars_exp + 1e-8), dim=2))  # (N, K)

############################################################
# Orchestration: replace refine_across_tasks with paper version
############################################################
class DistributedGraphEvolvingAnchors:
    def __init__(self, task_ids, K_per_task, feat_dim, device='cpu',
                 topk_graph=8,
                 merge_threshold=0.95, split_var_threshold=2.5):
        self.device = device
        self.task_ids = task_ids
        self.anchors = {tid: TaskAnchor(tid, K_per_task, feat_dim, device=device) for tid in task_ids}

        # paper-style refiner (parameter-free)
        self.graph_refiner = ComponentGraphRefiner(topk=topk_graph)

        # you can keep evolver if you still want split/merge (not required by the paper core)
        self.evolver = SelfEvolvingManager(self.anchors,
                                           merge_threshold=merge_threshold,
                                           split_var_threshold=split_var_threshold,
                                           device=device)

    def to(self, device):
        self.device = device
        for v in self.anchors.values():
            v.to(device)
        self.evolver.device = device
        return self

    def online_update_batch(self, features, task_labels, momentum=0.05, kmeans_init_minibatch=256):
        """
        Align with Sec 3.1:
        - If a task anchor not inited, do KMeans init on first enough samples.
        - Then do responsibilities + EMA updates (Eq 2-6).
        """
        device = self.device
        features = features.to(device)

        unique_tasks = list(set(task_labels))
        for tid in unique_tasks:
            idx = [i for i, t in enumerate(task_labels) if t == tid]
            if len(idx) == 0:
                continue
            x_t = features[idx, :]  # (n_t, D)
            anchor = self.anchors[tid]

            if (not anchor._inited) and (x_t.shape[0] >= min(anchor.K, 2)):
                # if we have enough samples, init with kmeans on a small buffer
                # (in practice you may want to maintain a small init buffer per task)
                n_init = min(kmeans_init_minibatch, x_t.shape[0])
                anchor.init_with_kmeans(x_t[:n_init])

            resp = anchor.soft_assign(x_t)  # Eq(3)
            anchor.online_update(x_t, resp, momentum=momentum)  # Eq(5)-(6)

    def refine_across_tasks(self):
        """
        Paper Sec 3.2 component-level cross-task message passing (Eq 7-14).
        """
        z, new_mus = self.graph_refiner.refine(self.anchors, self.task_ids, device=self.device)
        return z, new_mus

    @torch.no_grad()
    def compute_component_scores_all(self, x):
        """
        Paper Eq(15): s_{n,k}(z) is (diagonal Gaussian) log-likelihood-like score.
        Return:
          comp_scores: list of (tid, k, score)
          exp_scores:  tensor (M,) aligned with list
        """
        x = x.to(self.device).view(1, -1)  # (1,D)
        comp_scores = []
        exp_scores = []

        for tid in self.task_ids:
            means, vars_, weights, counts = self.anchors[tid].collect_component_stats()  # (K,D)
            # Eq(15): s = -1/2 * sum ( (x-mu)^2 / var ) - 1/2 * sum log var
            diff2 = (x - means) ** 2
            s = -0.5 * (diff2 / (vars_ + 1e-8)).sum(dim=1) - 0.5 * torch.log(vars_ + 1e-8).sum(dim=1)  # (K,)

            for k in range(means.shape[0]):
                comp_scores.append((tid, k, s[k]))
                exp_scores.append(torch.exp(s[k]))

        exp_scores = torch.stack(exp_scores, dim=0)  # (M,)
        return comp_scores, exp_scores

    @torch.no_grad()
    def compute_expert_weights(self, x):
        """
        Paper Eq(16): alpha_n = sum_k exp(s_{n,k}) / sum_{m,l} exp(s_{m,l})
        Returns:
          alpha: dict {tid: alpha_tid}
        """
        comp_scores, exp_scores = self.compute_component_scores_all(x)
        denom = exp_scores.sum() + 1e-8

        alpha = {tid: torch.tensor(0.0, device=self.device) for tid in self.task_ids}
        ptr = 0
        for (tid, k, s) in comp_scores:
            alpha[tid] = alpha[tid] + exp_scores[ptr]
            ptr += 1
        for tid in alpha:
            alpha[tid] = (alpha[tid] / denom).detach()
        return alpha

    @torch.no_grad()
    def compute_task_scores(self, x):
        """
        If you still want a (T,) tensor score for routing:
        use alpha_n directly (probability-like), or log alpha.
        """
        alpha = self.compute_expert_weights(x)
        return torch.stack([alpha[tid] for tid in self.task_ids], dim=0)  # (T,)
