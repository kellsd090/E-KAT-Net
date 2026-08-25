import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import umap
import hdbscan
import copy
import time
from thop import clever_format
from thop import profile as thop_profile
from sklearn.preprocessing import StandardScaler
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.collections import LineCollection
from torch.profiler import profile, ProfilerActivity


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SplineWeightLayer(nn.Module): # neurone cell body
    def __init__(self, in_features=187, in_hidden=64, out_hidden=64, kernel_size=15):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)
        self.alpha = nn.Parameter(torch.ones(1))  # I
        self.beta = nn.Parameter(torch.ones(1))  # W1 (Attention)
        self.theta = nn.Parameter(torch.ones(1))  # W3 (Convolution)
        self.gamma = nn.Parameter(torch.ones(1))  # residual
        self.W3_conv = nn.Conv1d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False)
        self.register_buffer('identity', torch.eye(in_features).unsqueeze(0))
        if in_hidden != out_hidden:
            self.residual_proj = nn.Linear(in_hidden, out_hidden)
        else:
            self.residual_proj = nn.Identity()
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x, W1=None):
        b, seq_len, h_dim = x.shape
        x_norm = self.ln(x)
        x_prime = torch.matmul(x_norm, self.W2) + self.b
        x_temporal = x_prime.transpose(1, 2)
        w3_feat = self.W3_conv(x_temporal).transpose(1, 2)
        if W1 is not None:
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * self.identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat
        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):   # edge
    def __init__(self, in_neurons, out_neurons, grid_size=64, spline_order=3, hidden_dim=64, neuron_out_dim=64,
                 kernel_size=15, seq_len=187):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_features=seq_len, in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size)
            for _ in range(out_neurons)])
        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))
        self.last_W1 = None
        with torch.no_grad():
            lin_init = torch.linspace(-1.0, 1.0, self.num_coeffs).to(self.spline_weights.device)
            initial_weights = lin_init.view(1, 1, -1).repeat(self.in_neurons, self.out_neurons, 1)
            self.spline_weights.data = initial_weights + torch.randn_like(initial_weights) * 0.01

    def b_spline_cubic(self, x):   # Kolmogorov-Arnold b-spline
        h = 2.0 / (self.num_coeffs - 1)
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res_outer = torch.relu(2.0 - dist) ** 3 / 6.0
        res_inner = torch.relu(1.0 - dist) ** 3 * (4.0 / 6.0)
        res = res_outer - res_inner
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape # x_in: [B, in_n, 200, 128]
        max_val = torch.max(torch.abs(x_in)) + 1e-8
        x_norm = (x_in / max_val) * 0.95
        x_norm = torch.clamp(x_norm, -0.99, 0.99)
        basis = self.b_spline_cubic(x_norm)
        spline_mapping = torch.einsum('binfc,ioc->bio nf', basis, self.spline_weights)
        omega_expanded = torch.abs(self.omiga).view(1, in_n, self.out_neurons, 1, 1)
        activated_signals = spline_mapping + omega_expanded * x_in.unsqueeze(2)
        next_outputs, next_projections = [], []
        t_val = torch.abs(self.temperature) * np.sqrt(seq_len / 256.0) + 1e-4
        
        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(current_edges ** 2, dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((torch.sqrt(edge_energies + 1e-8) - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)
            W1_j = None
            
            if proj_x_prev is not None:
                multiplier = (torch.sqrt(edge_energies + 1e-8) / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, 2*i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, 2*i+1, :, :] for i in range(mid)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            combined_input = torch.sum(current_edges * mask, dim=1)
            out_j, proj_j = self.weight_layers[j](combined_input, W1=W1_j)
            next_outputs.append(out_j)
            next_projections.append(proj_j)
        return torch.stack(next_outputs, dim=1), torch.stack(next_projections, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScalePrepConv(nn.Module):   # Conv layer before entering main model
    def __init__(self):
        super(MultiScalePrepConv, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), # 1 layer 64 channels
            nn.BatchNorm1d(16),
            nn.SiLU())
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=15, padding=7), 
            nn.BatchNorm1d(40),
            nn.SiLU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=25, padding=12), 
            nn.BatchNorm1d(8),
            nn.SiLU())

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return out # self.se(out)


class ECGKANModel(nn.Module):   # main E-KAT model
    def __init__(self, grid_size=64, spline_order=3, seq_len=187):
        super(ECGKANModel, self).__init__()
        self.prep_conv = MultiScalePrepConv()
        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order, hidden_dim=64, neuron_out_dim=64, kernel_size=25, seq_len=seq_len)
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order, hidden_dim=64, neuron_out_dim=128, kernel_size=15, seq_len=seq_len)
        self.layer3 = FastKANLayer(8, 8, grid_size, spline_order, hidden_dim=128, neuron_out_dim=32, kernel_size=9, seq_len=seq_len)
        self.layer4 = FastKANLayer(8, 5, grid_size, spline_order, hidden_dim=32, neuron_out_dim=1, kernel_size=3, seq_len=seq_len)

    def forward(self, x, return_latent=False):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.prep_conv(x)  # [B, 32, 187]
        x = x.transpose(1, 2).unsqueeze(1)
        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)  # x: [B, 5, 187, 1]
        if return_latent: return x.squeeze(-1)
        return torch.mean(x, dim=(-1, -2))

    def get_active_conn_info(self):
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.omiga)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {
                        "active": active_count,
                        "total": m.in_neurons * m.out_neurons,
                        "tau_mean": torch.abs(m.tau).mean().item(),
                        "ratio": active_count / (m.in_neurons * m.out_neurons + 1e-8)}
        return info


class PrunedSplineWeightLayer(nn.Module):
    def __init__(
        self,
        in_features=187,
        in_hidden=64,
        out_hidden=64,
        kernel_size=15,
        remove_alpha=False,
        remove_beta=False,
        remove_theta=False,
        remove_gamma=False
    ):
        super().__init__()
        self.in_features = in_features
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size
        self.remove_alpha = remove_alpha
        self.remove_beta = remove_beta
        self.remove_theta = remove_theta
        self.remove_gamma = remove_gamma
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)
        nn.init.xavier_uniform_(self.W2)

        if not remove_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("alpha", None)
        if not remove_beta:
            self.beta = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("beta", None)

        if not remove_theta:
            self.theta = nn.Parameter(torch.ones(1))
            self.W3_conv = nn.Conv1d(
                out_hidden,
                out_hidden,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=out_hidden,
                bias=False)
        else:
            self.register_parameter("theta", None)
            self.W3_conv = None

        if not remove_gamma:
            self.gamma = nn.Parameter(torch.ones(1))
            if in_hidden != out_hidden:
                self.residual_proj = nn.Linear(
                    in_hidden,
                    out_hidden)
            else:
                self.residual_proj = nn.Identity()
        else:
            self.register_parameter("gamma", None)
            self.residual_proj = None

    def forward(self, x, W1=None):
        x_norm = self.ln(x)
        x_prime = (torch.matmul(x_norm, self.W2)+ self.b)
        output = None
        if (
            not self.remove_beta
            and W1 is not None
        ):
            attention_part = (
                torch.abs(self.beta)
                * torch.bmm(W1, x_prime))
            output = attention_part

        if not self.remove_alpha:
            identity_part = (torch.abs(self.alpha)* x_prime)
            output = (
                identity_part
                if output is None
                else output + identity_part)

        if not self.remove_theta:
            x_temporal = x_prime.transpose(1, 2)
            w3_feat = self.W3_conv(
                x_temporal
            ).transpose(1, 2)
            conv_part = (
                torch.abs(self.theta)
                * w3_feat)
            output = (
                conv_part
                if output is None
                else output + conv_part)

        if not self.remove_gamma:
            res_x = self.residual_proj(x)
            residual_part = (
                self.gamma * res_x)
            output = (
                residual_part
                if output is None
                else output + residual_part)
          
        if output is None:
            raise RuntimeError("All neuron branches were removed.")
        return output, x_prime


class PrunedFastKANLayer(FastKANLayer):
    def __init__(
        self,
        in_neurons,
        out_neurons,
        grid_size=64,
        spline_order=3,
        hidden_dim=64,
        neuron_out_dim=64,
        kernel_size=15,
        seq_len=187,
        remove_alpha=False,
        remove_beta=False,
        remove_theta=False,
        remove_gamma=False
    ):
        super().__init__(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            grid_size=grid_size,
            spline_order=spline_order,
            hidden_dim=hidden_dim,
            neuron_out_dim=neuron_out_dim,
            kernel_size=kernel_size,
            seq_len=seq_len
        )
        self.remove_beta = remove_beta
      
        # Replace original neuron modules by physically
        self.weight_layers = nn.ModuleList([
            PrunedSplineWeightLayer(
                in_features=seq_len,
                in_hidden=hidden_dim,
                out_hidden=neuron_out_dim,
                kernel_size=kernel_size,
                remove_alpha=remove_alpha,
                remove_beta=remove_beta,
                remove_theta=remove_theta,
                remove_gamma=remove_gamma)
            for _ in range(out_neurons)
        ])


    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape

        # Edge-wise nonlinear mapping
        max_val = (torch.max(torch.abs(x_in)) + 1e-8)
        x_norm = (x_in / max_val ) * 0.95
        x_norm = torch.clamp(x_norm, -0.99, 0.99)
        basis = self.b_spline_cubic(x_norm)
        spline_mapping = torch.einsum(
            'binfc,ioc->bionf',
            basis,
            self.spline_weights)
        omega_expanded = (torch.abs(self.omiga)
            .view(1, in_n, self.out_neurons, 1, 1))
        activated_signals = (
            spline_mapping
            + omega_expanded
            * x_in.unsqueeze(2))
        next_outputs = []
        next_projections = []
        t_val = (
            torch.abs(self.temperature)
            * np.sqrt(seq_len / 256.0)
            + 1e-4)

        # Every output neuron
        for j in range(self.out_neurons):
            current_edges = (
                activated_signals[:, :, j, :, :])

            # Edge energy + soft gating remains unchanged
            edge_energies = torch.mean(
                current_edges ** 2,
                dim=(-1, -2))
            tau_j = (
                torch.abs(self.tau[:, j])
                .unsqueeze(0))
            energy_sqrt = torch.sqrt(
                edge_energies + 1e-8)
            mask = torch.sigmoid((energy_sqrt - tau_j) / t_val)
            mask_expanded = (mask.unsqueeze(-1).unsqueeze(-1))

            # Cross-layer attention
            W1_j = None
            if (
                proj_x_prev is not None
                and not self.remove_beta
            ):
                multiplier = (
                    energy_sqrt
                    / (tau_j + 1e-8))
                multiplier = (
                    multiplier
                    .unsqueeze(-1)
                    .unsqueeze(-1))
                weighted_prev = (
                    proj_x_prev
                    * multiplier
                    * mask_expanded)
                mid = self.in_neurons // 2
                K = torch.cat(
                    [weighted_prev[:, 2*i, :, :]
                        for i in range(mid)],dim=-1)
                Q = torch.cat(
                    [weighted_prev[:, 2*i+1, :, :]
                        for i in range(mid)], dim=-1)
                K = F.layer_norm(K, [K.size(-1)])
                Q = F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(
                    raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            # Gated edge aggregation
            combined_input = torch.sum(
                current_edges
                * mask_expanded,
                dim=1)

            out_j, proj_j = (
                self.weight_layers[j](
                    combined_input,
                    W1=W1_j))
            next_outputs.append(out_j)
          
            # beta removed -> projection does not need to be propagated for future attention
            if not self.remove_beta:
                next_projections.append(proj_j)
        output = torch.stack(
            next_outputs,
            dim=1)
        if self.remove_beta:
            projection = None
        else:
            projection = torch.stack(
                next_projections,
                dim=1)
        return output, projection


class SplineWeightLayerProfile(SplineWeightLayer):
    def __init__(
        self,
        *args,
        disable_alpha=False,
        disable_beta=False,
        disable_theta=False,
        disable_gamma=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.disable_alpha = disable_alpha
        self.disable_beta = disable_beta
        self.disable_theta = disable_theta
        self.disable_gamma = disable_gamma

    def forward(self, x, W1=None):
        x_norm = self.ln(x)
        x_prime = torch.matmul(x_norm, self.W2) + self.b
        if not self.disable_theta:
            x_temporal = x_prime.transpose(1, 2)
            w3_feat = self.W3_conv(x_temporal).transpose(1, 2)
        else:
            w3_feat = None
        spatial_combined = 0.0
      
        if (not self.disable_beta) and (W1 is not None):
            spatial_combined = (
                spatial_combined
                + torch.abs(self.beta) * torch.bmm(W1, x_prime))

        if not self.disable_alpha:
            spatial_combined = (
                spatial_combined
                + torch.abs(self.alpha) * x_prime)

        if not self.disable_theta:
            spatial_combined = (
                spatial_combined
                + torch.abs(self.theta) * w3_feat)

        if not self.disable_gamma:
            res_x = self.residual_proj(x)
            spatial_combined = (
                spatial_combined
                + self.gamma * res_x)

        return spatial_combined, x_prime


class FastKANLayerProfile(FastKANLayer):
    def __init__(
        self,
        *args,
        gating_mode="soft_learned",
        fixed_temperature=1.0,
        disable_alpha=False,
        disable_beta=False,
        disable_theta=False,
        disable_gamma=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gating_mode = gating_mode
        self.fixed_temperature = fixed_temperature

        # Replace original weight layers by profiling layers
        old_layers = self.weight_layers
        new_layers = []
        for old in old_layers:
            new = SplineWeightLayerProfile(
                in_features=old.in_features,
                in_hidden=old.in_hidden,
                out_hidden=old.out_hidden,
                kernel_size=old.k,
                disable_alpha=disable_alpha,
                disable_beta=disable_beta,
                disable_theta=disable_theta,
                disable_gamma=disable_gamma)
            new.load_state_dict(old.state_dict(), strict=False)
            new_layers.append(new)
        self.weight_layers = nn.ModuleList(new_layers)

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape
        max_val = torch.max(torch.abs(x_in)) + 1e-8
        x_norm = (x_in / max_val) * 0.95
        x_norm = torch.clamp(x_norm, -0.99, 0.99)
        basis = self.b_spline_cubic(x_norm)
        spline_mapping = torch.einsum(
            'binfc,ioc->bionf',
            basis,
            self.spline_weights)
        omega_expanded = torch.abs(self.omiga).view(1, in_n,
            self.out_neurons, 1, 1)
        activated_signals = (
            spline_mapping
            + omega_expanded * x_in.unsqueeze(2))
        next_outputs = []
        next_projections = []

        # Temperature
        if self.gating_mode == "soft_fixed":
            t_val = (
                self.fixed_temperature
                * np.sqrt(seq_len / 256.0)
                + 1e-4)
        elif self.gating_mode == "soft_learned":
            t_val = (
                torch.abs(self.temperature)
                * np.sqrt(seq_len / 256.0)
                + 1e-4)
        else:
            # hard / none do not require T
            t_val = None
        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]

            # Soft / hard gating requires edge energy
            if self.gating_mode != "none":
                edge_energies = torch.mean(
                    current_edges ** 2,
                    dim=(-1, -2))
              
                tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
                energy_sqrt = torch.sqrt(edge_energies + 1e-8)

            # Soft gating
            if self.gating_mode in [
                "soft_fixed",
                "soft_learned"
            ]:
                mask = torch.sigmoid((energy_sqrt - tau_j) / t_val)

            # Hard gating
            elif self.gating_mode == "hard":
                mask = (energy_sqrt > tau_j).to(current_edges.dtype)

            # No gating
            elif self.gating_mode == "none":
                mask = torch.ones( b, in_n,
                    device=x_in.device,
                    dtype=x_in.dtype)
            else:
                raise ValueError(
                    f"Unknown gating mode: "
                    f"{self.gating_mode}")

            mask_expanded = (
                mask
                .unsqueeze(-1)
                .unsqueeze(-1))

            # Cross-layer attention
            W1_j = None
            if proj_x_prev is not None:

                if self.gating_mode in [
                    "soft_fixed",
                    "soft_learned"
                ]:
                    multiplier = (energy_sqrt / (tau_j + 1e-8))
                    weighted_prev = (
                        proj_x_prev
                        * multiplier.unsqueeze(-1).unsqueeze(-1)
                        * mask_expanded)
                elif self.gating_mode == "hard":
                    weighted_prev = (proj_x_prev * mask_expanded)
                else: 
                    weighted_prev = proj_x_prev
                mid = self.in_neurons // 2
                K = torch.cat(
                    [
                        weighted_prev[:, 2*i, :, :]
                        for i in range(mid)
                    ],dim=-1 )

                Q = torch.cat(
                    [weighted_prev[:, 2*i+1, :, :]
                        for i in range(mid)], dim=-1)

                K = F.layer_norm(K, [K.size(-1)])
                Q = F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))

                if self.gating_mode in [
                    "soft_fixed",
                    "soft_learned"]:

                    W1_j = F.softmax(
                        raw_attn / ( np.sqrt(K.shape[-1]) * t_val),dim=-1)
                else:
                    W1_j = F.softmax(
                        raw_attn / np.sqrt(K.shape[-1]), dim=-1)

            # Edge aggregation
            combined_input = torch.sum(
                current_edges * mask_expanded, dim=1)
            out_j, proj_j = self.weight_layers[j](
                combined_input, W1=W1_j)
            next_outputs.append(out_j)
            next_projections.append(proj_j)
            
        return (torch.stack(next_outputs, dim=1),
            torch.stack(next_projections, dim=1))


class ECGKANModelProfile(nn.Module):
    def __init__(
        self,
        grid_size=64,
        spline_order=3,
        seq_len=187,
        ablate=None,
        gating_mode="soft_learned",
        fixed_temperature=1.0
    ):
        super().__init__()
        self.prep_conv = MultiScalePrepConv()
        disable_alpha = (ablate == "alpha")
        disable_beta  = (ablate == "beta")
        disable_theta = (ablate == "theta")
        disable_gamma = (ablate == "gamma")
        common = dict(
            grid_size=grid_size,
            spline_order=spline_order,
            seq_len=seq_len,
            gating_mode=gating_mode,
            fixed_temperature=fixed_temperature,
            disable_alpha=disable_alpha,
            disable_beta=disable_beta,
            disable_theta=disable_theta,
            disable_gamma=disable_gamma
        )

        self.layer1 = FastKANLayerProfile(
            1, 4,
            hidden_dim=64,
            neuron_out_dim=64,
            kernel_size=25,
            **common
        )

        self.layer2 = FastKANLayerProfile(
            4, 8,
            hidden_dim=64,
            neuron_out_dim=128,
            kernel_size=15,
            **common
        )

        self.layer3 = FastKANLayerProfile(
            8, 8,
            hidden_dim=128,
            neuron_out_dim=32,
            kernel_size=9,
            **common
        )

        self.layer4 = FastKANLayerProfile(
            8, 5,
            hidden_dim=32,
            neuron_out_dim=1,
            kernel_size=3,
            **common
        )


    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.prep_conv(x)
        x = (x.transpose(1, 2) .unsqueeze(1))
        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)
        return torch.mean(x, dim=(-1, -2))


class ECGKANPrunedModel(nn.Module):
    def __init__(
        self,
        ablate=None,
        grid_size=64,
        spline_order=3,
        seq_len=187
    ):
        super().__init__()
        self.ablate = ablate
        self.prep_conv = MultiScalePrepConv()
        remove_alpha = (ablate == "alpha")
        remove_beta = (ablate == "beta")
        remove_theta = (ablate == "theta")
        remove_gamma = (ablate == "gamma")
        common = dict(
            grid_size=grid_size,
            spline_order=spline_order,
            seq_len=seq_len,
            remove_alpha=remove_alpha,
            remove_beta=remove_beta,
            remove_theta=remove_theta,
            remove_gamma=remove_gamma)
        self.layer1 = PrunedFastKANLayer(
            1, 4, hidden_dim=64,
            neuron_out_dim=64,
            kernel_size=25, **common)
        self.layer2 = PrunedFastKANLayer(
            4, 8, hidden_dim=64,
            neuron_out_dim=128,
            kernel_size=15, **common)
        self.layer3 = PrunedFastKANLayer(
            8, 8, hidden_dim=128,
            neuron_out_dim=32,
            kernel_size=9, **common)
        self.layer4 = PrunedFastKANLayer(
            8, 5, hidden_dim=32,
            neuron_out_dim=1,
            kernel_size=3, **common)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.prep_conv(x)
        x = (x.transpose(1, 2).unsqueeze(1))
        x, proj1 = self.layer1(x, proj_x_prev=None)
        if self.ablate == "beta":
            x, _ = self.layer2(x, proj_x_prev=None)
            x, _ = self.layer3(x, proj_x_prev=None)
            x, _ = self.layer4( x, proj_x_prev=None)
        else:
            x, proj2 = self.layer2(x, proj_x_prev=proj1)
            x, proj3 = self.layer3(x, proj_x_prev=proj2)
            x, proj4 = self.layer4(x, proj_x_prev=proj3)
        return torch.mean(x, dim=(-1, -2))


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
weight_path = "/content/parameter.pth" 


def build_profile_model(
    ablate=None,
    gating_mode="soft_learned",
    fixed_temperature=1.0):
    model = ECGKANModelProfile(
        ablate=ablate,
        gating_mode=gating_mode,
        fixed_temperature=fixed_temperature
    ).to(device)
    state_dict = torch.load(
        weight_path,
        map_location=device)
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False)
    model.eval()
    return model


def count_model_params(model):
    total = sum(
        p.numel()
        for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad)
    return total, trainable


def freeze_temperature(model):
    for m in model.modules():
        if isinstance(m, FastKANLayerProfile):
            m.temperature.requires_grad = False


def format_number(n):
    if n >= 1e9:
        return f"{n / 1e9:.4f} G"
    elif n >= 1e6:
        return f"{n / 1e6:.4f} M"
    elif n >= 1e3:
        return f"{n / 1e3:.4f} K"
    else:
        return str(n)


def build_pruned_model(ablate=None):
    model = ECGKANPrunedModel(
        ablate=ablate).to(device)
    state_dict = torch.load(
        weight_path,
        map_location=device)
    result = model.load_state_dict(
        state_dict,
        strict=False)
    model.eval()
    return model


def calculate_model_macs(model, device):
    model.eval()
    dummy_input = torch.randn(1, 187, device=device)
    with torch.no_grad():
        macs, _ = thop_profile(
            model,
            inputs=(dummy_input,),
            verbose=False)
    return macs


# Params and MACs while α,β,θ,γ four modules are removed respectively
settings = {
    "alpha = 0": "alpha",
    "gamma = 0": "gamma",
    "theta = 0": "theta",
    "beta = 0": "beta",
    "Original": None}
results = []

for name, ablation in settings.items():
    model = build_pruned_model(ablate=ablation)
    total_params, trainable_params = (count_model_params(model))
    macs = calculate_model_macs(model, device)
    results.append({
        "Setting": name,
        "Params": total_params,
        "Trainable": trainable_params,
        "MACs": macs})
    print(
        f"{name:12s} | "
        f"Params: "
        f"{total_params / 1:.1f}  | "
        f"Trainable: "
        f"{trainable_params / 1e6:.4f} M | "
        f"MACs: "
        f"{macs / 1e9:.4f} G")


# Params and MACs while soft gating (specific T), soft gating (trained T), hard gate, no gate are used respectively
def count_effective_gating_params(model, mode):
    total = sum(p.numel() for p in model.parameters())
    removed = 0
    for m in model.modules():
        if isinstance(m, FastKANLayerProfile):
            if mode in ["soft_fixed", "hard"]:
                if hasattr(m, "temperature") and m.temperature is not None:
                    removed += m.temperature.numel()
            elif mode == "none":
                if hasattr(m, "temperature") and m.temperature is not None:
                    removed += m.temperature.numel()
                if hasattr(m, "tau") and m.tau is not None:
                    removed += m.tau.numel()
    effective_params = total - removed
    return effective_params

gating_settings = {
    "Soft, T = 1": "soft_fixed",
    "Hard": "hard",
    "None": "none",
    "Soft, T = learned": "soft_learned"}

for name, mode in gating_settings.items():
    model = build_profile_model(
        ablate=None,
        gating_mode=mode,
        fixed_temperature=1.0)
    params = count_effective_gating_params(model, mode)
    macs = calculate_model_macs(model, device)
    print(
        f"{name:22s} | "
        f"Params: {params:,} | "
        f"MACs: {macs/1e9:.4f} G")
