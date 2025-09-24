import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from collections import OrderedDict
from random import shuffle, sample
from time import perf_counter

import numpy as np
import torch
import torchvision.transforms as transforms

# universal optimizer class
class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=0.02, num=0.5, kmult=0.007, name=None, device=None, 
                amplifier=.1, theta=np.inf, damping=.4, eps=1e-5, weight_decay=0, kappa=0.9, stratified=True):

        defaults = dict(idx=idx, lr=lr, w=w, num=num, kmult=kmult, agents=agents, name=name, device=device,
                amplifier=amplifier, theta=theta, damping=damping, eps=eps, weight_decay=weight_decay, kappa=kappa, lamb=lr, stratified=stratified)
        
        super(Base, self).__init__(params, defaults)

    # grad and param difference norms
    def compute_dif_norms(self, prev_optimizer):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data
                grad_dif_norm += (d_p - prev_d_p).norm().item() ** 2
                param_dif_norm += (p.data - prev_p.data).norm().item() ** 2

            gr, pra = np.sqrt(grad_dif_norm), np.sqrt(param_dif_norm)
            group['grad_dif_norm'] = np.sqrt(grad_dif_norm)
            group['param_dif_norm'] = np.sqrt(param_dif_norm)
        return gr, pra

    # compute gradient differences`
    def compute_grad_dif(self, prev_optimizer):
        grad_diffs = []  # 存储所有参数的梯度差值张量

        # 遍历每个参数组
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            # 为当前参数组初始化梯度差值列表
            group_grad_diffs = []

            # 遍历组内每个参数
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    group_grad_diffs.append(None)  
                    continue

                grad_diff = p.grad.data - prev_p.grad.data
                group_grad_diffs.append(grad_diff.clone())  # 深拷贝避免后续修改

            # 将当前组的梯度差值存入参数组字典
            group['grad_differences'] = group_grad_diffs
            grad_diffs.extend(group_grad_diffs)

        return grad_diffs  # 返回所有参数的梯度差值列表

    #weird one
    def set_norms(self, grad_diff, param_diff):
        for group in self.param_groups:
            group['grad_dif_norm'] = grad_diff
            group['param_dif_norm'] = param_diff
    
    def collect_params(self, lr=False):
        for group in self.param_groups:
            grads = []
            vars = []
            if lr:
                return group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                vars.append(p.data.clone().detach())
                grads.append(p.grad.data.clone().detach())
        return vars, grads
    
    def step(self):
        pass
            

class Fedrcu(Base):
    """
    Implements Algorithm 1 FedRecu for a single client i.

    Notation (matching your figure):
      - x_i(t)   : current params      -> group['params'][k].data
      - x_i(t-1) : previous params     -> group['prev_params'][k]
      - ∇f_i(x_t): current grads       -> group['params'][k].grad
      - ∇f_i(x_{t-1}): previous grads  -> group['prev_grads'][k]

    Public API (call these on each client):
      - transmit_v():   returns v_i(t) list of tensors         [Eq. (3)]
      - apply_avg_v(avg_v): sets x_i(t+1)= (1/N)∑ v_j(t)       [Eq. (4)]
      - transmit_w():   returns w_i(t) list of tensors         [Eq. (5)]
      - apply_avg_w(avg_w): sets x_i(t+1)= 2 x_i(t) - avg_w    [Eq. (6)]
      - local_step():   x_i(t+1)= 2x_i(t)-x_i(t-1)-α(∇f_t-∇f_{t-1})  [Eq. (7)]

    IMPORTANT:
      - Before calling any of the methods that use previous state, make sure
        you have already run one forward/backward so that current grads exist.
      - After *every* state-changing method (apply_avg_v / apply_avg_w / local_step),
        we cache (x_i(t), ∇f_i(x_t)) as the new "previous" for the next round.
    Internal 'previous' state is stored ONLY in:
        - group['prev_params']  == x_{t-1}
        - group['prev_grads']   == g_{t-1}

    Update pattern per iteration:
      1) (lazy) bootstrap prev_* with current (x, g) if first use.
      2) For local_step:
           - snapshot old_prev = (prev_params, prev_grads)
           - set prev_* <- (x_t, g_t) BEFORE mutating params
           - compute x_{t+1} using old_prev and current (x_t, g_t)
         For apply_avg_v / apply_avg_w:
           - set prev_* <- (x_t, g_t) BEFORE assigning the new state
           - assign x_{t+1}
      3) Do NOT touch prev_* again until next iteration.
    """
    def __init__(self, *args, **kwargs):
        super(Fedrcu, self).__init__(*args, **kwargs)
        self._bootstrapped = False
        for group in self.param_groups:
            n = len(group['params'])
            group.setdefault('prev_params', [None]*n)
            group.setdefault('prev_grads',  [None]*n)

    # ---------- internals ----------
    @torch.no_grad()
    def _bootstrap_if_needed(self):
        if self._bootstrapped:
            return
        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                group['prev_params'][k] = p.data.clone()
                if p.grad is None:
                    group['prev_grads'][k] = torch.zeros_like(p.data)
                else:
                    group['prev_grads'][k] = p.grad.data.clone()
        self._bootstrapped = True

    @torch.no_grad()
    def _write_prev_from_current(self):
        """prev_* <- (x_t, g_t) BEFORE mutating params."""
        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                group['prev_params'][k] = p.data.clone()
                if p.grad is None:
                    group['prev_grads'][k] = torch.zeros_like(p.data)
                else:
                    group['prev_grads'][k] = p.grad.data.clone()

    # ---------- payloads (no mutation) ----------
    @torch.no_grad()
    def transmit_v(self):
        """
        v_i(t) = 2 x(t) - x(t-1) - α g(t) + α g(t-1)
        """
        self._bootstrap_if_needed()
        outs = []
        for group in self.param_groups:
            lr = group['lr']
            for k, p in enumerate(group['params']):
                x_t   = p.data
                g_t   = torch.zeros_like(x_t) if p.grad is None else p.grad.data
                x_tm1 = group['prev_params'][k]
                g_tm1 = group['prev_grads'][k]
                v_k = 2 * x_t - x_tm1 - lr * g_t + lr * g_tm1
                outs.append(v_k.clone())
        return outs

    @torch.no_grad()
    def transmit_w(self):
        """
        w_i(t) = x(t-1) + α g(t) - α g(t-1)
        """
        self._bootstrap_if_needed()
        outs = []
        for group in self.param_groups:
            lr = group['lr']
            for k, p in enumerate(group['params']):
                g_t   = torch.zeros_like(p.data) if p.grad is None else p.grad.data
                x_tm1 = group['prev_params'][k]
                g_tm1 = group['prev_grads'][k]
                w_k = x_tm1 + lr * g_t - lr * g_tm1
                outs.append(w_k.clone())
        return outs

    # ---------- state updates (mutation) ----------
    @torch.no_grad()
    def apply_avg_v(self, avg_v_flat):
        """
        End of a v-round: x(t+1) <- avg_v.
        prev_* is set to (x_t, g_t) BEFORE assignment so next iteration sees (t) as previous.
        """
        self._bootstrap_if_needed()
        self._write_prev_from_current()
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(avg_v_flat[i])
                i += 1

    @torch.no_grad()
    def apply_avg_w(self, avg_w_flat):
        """
        End of a w-round: x(t+1) <- 2 x(t) - avg_w.
        prev_* is set to (x_t, g_t) BEFORE assignment.
        """
        self._bootstrap_if_needed()
        self._write_prev_from_current()
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                x_t = p.data
                p.data.copy_(2 * x_t - avg_w_flat[i])
                i += 1

    @torch.no_grad()
    def local_step(self):
        """
        Local-only:
          x(t+1) = 2 x(t) - x(t-1) - α g(t) + α g(t-1)
        Uses current (x_t, g_t) and stored prev_* = (x_{t-1}, g_{t-1}).
        """
        self._bootstrap_if_needed()
        # Keep a local snapshot of previous before we overwrite prev_* with current.
        for group in self.param_groups:
            lr = group['lr']
            prev_x = [px.clone() for px in group['prev_params']]
            prev_g = [pg.clone() for pg in group['prev_grads']]

            # prev_* <- (x_t, g_t) BEFORE mutating
            for k, p in enumerate(group['params']):
                group['prev_params'][k] = p.data.clone()
                group['prev_grads'][k]  = torch.zeros_like(p.data) if p.grad is None else p.grad.data.clone()

            # compute update using old (x_{t-1}, g_{t-1}) and current (x_t, g_t)
            for k, p in enumerate(group['params']):
                x_t = group['prev_params'][k]   # equals p.data prior to mutation
                g_t = group['prev_grads'][k]
                x_tm1 = prev_x[k]
                g_tm1 = prev_g[k]
                p.data = 2 * x_t - x_tm1 - lr * g_t + lr * g_tm1

    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        self.local_step()
        return loss


class Scaffold(Base):
    """
    SCAFFOLD (Karimireddy et al., 2020)
    Local update:  x <- x - η * ( ∇f_i(x) - c + c_i )
      where c is global control variate, c_i is client control variate.

    At a communication:
      Let K be local steps since last comm, x0 = params at round start, xK = current params.
      c_i' = c_i - c + (1/(K*η)) (x0 - xK)
      Server: c <- c + (1/m) Σ_i (c_i' - c_i), then set each c_i <- c_i'.

    This class keeps all state internally:
      - group['c_global'][k] : c
      - group['c_local'][k]  : c_i
      - group['round_start_params'][k] : x0
      - group['c_snapshot'][k] : c at round start (for completeness; not strictly needed)
      - group['steps_since_sync'] : K
    """

    def __init__(self, *args, **kwargs):
        super(Scaffold, self).__init__(*args, **kwargs)
        for group in self.param_groups:
            n = len(group['params'])
            # control variates
            group.setdefault('c_global', [None]*n)
            group.setdefault('c_local',  [None]*n)
            # per-round bookkeeping
            group.setdefault('round_start_params', [None]*n)
            group.setdefault('c_snapshot',        [None]*n)
            group.setdefault('steps_since_sync',  0)

        # lazy bootstrap: fill zeros and mark round start as current state
        self._bootstrapped = False

    # ---------- internal helpers ----------

    @torch.no_grad()
    def _bootstrap_if_needed(self):
        if self._bootstrapped:
            return
        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                z = torch.zeros_like(p.data)
                group['c_global'][k] = z.clone()
                group['c_local'][k]  = z.clone()
                group['round_start_params'][k] = p.data.clone()
                group['c_snapshot'][k]        = z.clone()
            group['steps_since_sync'] = 0
        self._bootstrapped = True

    @torch.no_grad()
    def begin_round(self):
        """Call at the **start** of a communication round (after syncing params)."""
        self._bootstrap_if_needed()
        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                group['round_start_params'][k] = p.data.clone()
                group['c_snapshot'][k] = group['c_global'][k].clone()
            group['steps_since_sync'] = 0

    # ---------- local update ----------

    @torch.no_grad()
    def local_step(self):
        """
        One SCAFFOLD local step on this client's current minibatch:
          x <- x - lr * ( grad - c_global + c_local )
        """
        self._bootstrap_if_needed()
        for group in self.param_groups:
            lr = group['lr']
            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                c  = group['c_global'][k]
                ci = group['c_local'][k]
                p.data.add_( g - ci + c, alpha=-lr )
            group['steps_since_sync'] += 1

    # ---------- server payloads ----------

    @torch.no_grad()
    def transmit_params(self):
        """Return flat list of current params for (FedAvg-style) model averaging."""
        self._bootstrap_if_needed()
        out = []
        for group in self.param_groups:
            for p in group['params']:
                out.append(p.data.clone())
        return out

    @torch.no_grad()
    def transmit_c_delta(self):
        """
        Return Δc_i = c_i' - c_i  =  -c + (1/(K*lr)) * (x0 - xK)
        Computed per-parameter for this client since last comm.
        """
        self._bootstrap_if_needed()
        outs = []
        for group in self.param_groups:
            K = max(1, int(group['steps_since_sync']))
            lr = group['lr']
            for k, p in enumerate(group['params']):
                x0 = group['round_start_params'][k]
                xK = p.data
                c  = group['c_snapshot'][k]       # c used during the round
                delta = (-c) + (x0 - xK) / (K * lr)
                # sanitize just in case
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                outs.append(delta.clone())
        return outs

    # ---------- server applications ----------

    @torch.no_grad()
    def apply_global_params(self, avg_params_flat):
        """Set params to averaged model (FedAvg-style)."""
        self._bootstrap_if_needed()
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(avg_params_flat[i])
                i += 1

    @torch.no_grad()
    def apply_c_updates(self, avg_delta_c_flat, client_delta_c_flat):
        """
        Server tells:
          c <- c + avg_delta_c
          c_i <- c_i + client_delta_c
        """
        self._bootstrap_if_needed()
        gi = 0
        for group in self.param_groups:
            for k, _ in enumerate(group['params']):
                group['c_global'][k].add_(avg_delta_c_flat[gi])
                group['c_local'][k].add_(client_delta_c_flat[gi])
                gi += 1

    # (optional) make .step() do a local_step so your loops can call optimizer.step()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():  # standard pattern
                loss = closure()
        else:
            loss = None
        self.local_step()
        return loss


class FedLin(Base):
    """
    FedLin without compression or error-feedback.

    Local update on client i (per minibatch):
        x <- x - η * ( ∇f_i(x) - ∇f_i( x̄_t ) + g_t )

    Server state:
        g_t = (1/m) Σ_i ∇f_i( x̄_t )

    This optimizer keeps per-parameter lists on each client:
      - group['g']        : server vector g_t (broadcast each round)
      - group['grad_bar'] : client's ∇f_i( x̄_t ), set by trainer each step/round
    """

    def __init__(self, *args, **kwargs):
        super(FedLin, self).__init__(*args, **kwargs)
        for group in self.param_groups:
            n = len(group['params'])
            group.setdefault('g',        [torch.zeros_like(p.data) for p in group['params']])
            group.setdefault('grad_bar', [torch.zeros_like(p.data) for p in group['params']])

    # --- setters called by trainer ---

    @torch.no_grad()
    def set_g(self, g_flat):
        i = 0
        for group in self.param_groups:
            for k, _ in enumerate(group['params']):
                group['g'][k].copy_(g_flat[i])
                i += 1

    @torch.no_grad()
    def set_grad_bar(self, grad_bar_flat):
        i = 0
        for group in self.param_groups:
            for k, _ in enumerate(group['params']):
                group['grad_bar'][k].copy_(grad_bar_flat[i])
                i += 1

    # --- core local step ---

    @torch.no_grad()
    def local_step(self):
        """
        x <- x - lr * ( grad_local - grad_bar + g )
        Uses current p.grad as grad_local.
        """
        for group in self.param_groups:
            lr = group['lr']
            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                g_local = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                g_bar   = torch.nan_to_num(group['grad_bar'][k], nan=0.0, posinf=0.0, neginf=0.0)
                g_srv   = torch.nan_to_num(group['g'][k],       nan=0.0, posinf=0.0, neginf=0.0)
                p.data.add_(g_local - g_bar + g_srv, alpha=-lr)

    # --- payloads for server averaging ---

    @torch.no_grad()
    def transmit_params(self):
        out = []
        for group in self.param_groups:
            for p in group['params']:
                out.append(p.data.clone())
        return out


class Scaffnew(Base):
    """
    ProxSkip / Scaffnew (centralized).

    Per step on client i:
      xhat_i = x_i - γ (∇f_i(x_i) - h_i)
      if coin θ=1 (with prob p set by TRAINER):
          x_i ← avg_j xhat_j
      else:
          x_i ← xhat_i
      h_i ← h_i + (p/γ) (x_i - xhat_i)

    NOTE: p is NOT stored here. The trainer passes p into `apply_from_xhat(...)`.
    """

    def __init__(self, *args, **kwargs):
        super(Scaffnew, self).__init__(*args, **kwargs)
        for group in self.param_groups:
            n = len(group['params'])
            group.setdefault('h',    [torch.zeros_like(p.data) for p in group['params']])
            group.setdefault('xhat', [torch.zeros_like(p.data) for p in group['params']])

    @torch.no_grad()
    def transmit_xhat(self):
        """
        Build x̂ using current grads and store it; return flat list of x̂.
        x̂ = x - γ (grad - h)
        """
        outs = []
        for group in self.param_groups:
            gamma = float(group['lr'])
            for k, p in enumerate(group['params']):
                g = torch.zeros_like(p.data) if p.grad is None else torch.nan_to_num(
                    p.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                )
                h = group['h'][k]
                xhat = p.data - gamma * (g - h)
                group['xhat'][k] = xhat.clone()
                outs.append(xhat.clone())
        return outs

    @torch.no_grad()
    def apply_from_xhat(self, target_flat, p_comm):
        """
        Set parameters to `target_flat` (avg or own x̂) and update h:
          h ← h + (p/γ) (x - x̂)
        `p_comm` (probability p) is provided by the TRAINER.
        """
        i = 0
        for group in self.param_groups:
            gamma = float(group['lr'])
            scale = float(p_comm) / max(gamma, 1e-12)
            for k, p in enumerate(group['params']):
                target = target_flat[i]
                xhat_k = group['xhat'][k]
                p.data.copy_(target)
                group['h'][k].add_( (target - xhat_k) * scale )
                i += 1

    def step(self, closure=None):
        # the trainer orchestrates the sequence; no-op here
        return None
