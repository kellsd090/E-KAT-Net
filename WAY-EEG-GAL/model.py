import pandas as pd
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import toeplitz
import umap
import hdbscan
from thop import profile, clever_format
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.collections import LineCollection
from dataset_gal import WAYEEGDataset, calculate_pos_weights, visualize_waveforms


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


class SplineWeightLayer(nn.Module):   # neurone cell body
    def __init__(self, in_features=250, in_hidden=128, out_hidden=128, kernel_size=15):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # Identity
        self.beta = nn.Parameter(torch.ones(1))  # Attn
        self.theta = nn.Parameter(torch.ones(1))  # Conv
        self.gamma = nn.Parameter(torch.tensor([0.5]))  # Residual
        self.W3_conv = nn.Conv1d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False
        )
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
    def __init__(self, in_neurons, out_neurons, grid_size=128, spline_order=3, hidden_dim=128, neuron_out_dim=128,
                 kernel_size=15, seq_len=250):
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

    def b_spline_cubic(self, x):   # Kolmogorov Arnold b-spline
        h = 2.0 / (self.num_coeffs - 1)   # [B, in_n, S, D, 1] - [num_coeffs] -> [B, in_n, S, D, num_coeffs]
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res_outer = torch.relu(2.0 - dist) ** 3 / 6.0
        res_inner = torch.relu(1.0 - dist) ** 3 * (4.0 / 6.0)
        res = res_outer - res_inner
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape   # x_in: [B, in_n, 200, 128]
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


class BCIPrepConv(nn.Module):   # convolution before entering E-KAT main model
    def __init__(self, in_channels=160, out_channels=128, seq_len=200, stride=2):
        super(BCIPrepConv, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=5, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.SiLU())
        self.stride = stride
        mid_channels = out_channels    # 2 layers 128 channels
        self.branch_short = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 4, kernel_size=15, stride=self.stride, padding=7),
            nn.BatchNorm1d(mid_channels // 4), nn.SiLU())
        self.branch_mid = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 2, kernel_size=33, stride=self.stride, padding=16),
            nn.BatchNorm1d(mid_channels // 2), nn.SiLU())
        self.branch_long = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 4, kernel_size=65, stride=self.stride, padding=32),
            nn.BatchNorm1d(mid_channels // 4), nn.SiLU())
        
        self.post_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels))

        self.shortcut = nn.Sequential(    # # residual
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride),
            nn.BatchNorm1d(out_channels))

    def forward(self, x):   # x : [B, 160, 200]
        x_spatial = self.spatial_conv(x)
        out = torch.cat([
            self.branch_short(x_spatial),
            self.branch_mid(x_spatial),
            self.branch_long(x_spatial)], dim=1)
        res = self.shortcut(x)
        out = F.silu(self.post_conv(out) + res)
        # [B, Input_Neurons=1, New_Seq_Len, Feature_Dim=64]
        out = out.transpose(1, 2).unsqueeze(1)
        return out


class BCIKANModel(nn.Module):    # main E-KAT model
    def __init__(self, grid_size=64, spline_order=3, seq_len=125):
        super(BCIKANModel, self).__init__()
        self.prep_conv = BCIPrepConv(in_channels=160, out_channels=128, stride=2)   # [B, 3, 750] -> [B, 1, 187, 64]
        
        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=128, kernel_size=55, seq_len=seq_len)
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=128, kernel_size=33, seq_len=seq_len)
        self.layer3 = FastKANLayer(8, 8, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=64, kernel_size=17, seq_len=seq_len)
        self.layer4 = FastKANLayer(8, 12, grid_size, spline_order,
                                   hidden_dim=64, neuron_out_dim=32, kernel_size=9, seq_len=seq_len)
        self.layer5 = FastKANLayer(12, 8, grid_size, spline_order,
                                   hidden_dim=32, neuron_out_dim=16, kernel_size=5, seq_len=seq_len)
        self.layer6 = FastKANLayer(8, 6, grid_size, spline_order,
                                   hidden_dim=16, neuron_out_dim=6, kernel_size=3, seq_len=seq_len)

    def forward(self, x, return_latent=False):
        # x: [B, 160, 250] ->  [B, 1, 125, 128]
        x = self.prep_conv(x)
        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)
        x, proj5 = self.layer5(x, proj_x_prev=proj4)
        x, proj6 = self.layer6(x, proj_x_prev=proj5) 
        # x: [B, 6, 125, 6] ->  [B, 6, 125]
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
                        "ratio": active_count / (m.in_neurons * m.out_neurons)}
        return info


# visualization
def visualize_w1_analysis(layer, layer_idx, neuron_idx):
    W1_data = layer.last_W1[0].detach().cpu().numpy()
    col_ratio = np.sum(W1_data, axis=0) / (np.sum(W1_data) + 1e-8)
    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[20, 1], hspace=0.05, wspace=0.02)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(range(len(col_ratio)), col_ratio, color='skyblue', width=0.8)
    ax_bar.set_xlim(-0.5, len(col_ratio) - 0.5)
    ax_bar.set_ylabel("Attention Weight")
    ax_heat = fig.add_subplot(gs[1, 0])
    sns.heatmap(W1_data, ax=ax_heat, cmap='magma', cbar=False) 
    ax_cbar = fig.add_subplot(gs[1, 1])
    norm = plt.Normalize(W1_data.min(), W1_data.max())
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='magma'), cax=ax_cbar, label='Intensity')
    plt.show()
    return col_ratio


def visualize_spline_mapping_window(layer, layer_idx, neuron_idx, edge_idx):
    device = next(layer.parameters()).device
    fig, ax = plt.subplots(figsize=(8, 6))
    with torch.no_grad():
        x_range = torch.linspace(-1, 1, 200).to(device)
        current_omiga = torch.abs(layer.omiga[edge_idx[0], edge_idx[1]])
        basis = layer.b_spline_cubic(x_range.unsqueeze(0))
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        y_spline = torch.matmul(basis, coeffs).squeeze()
        y_linear = current_omiga * x_range
        y_final = y_spline + y_linear
        ax.plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline $f(x)$', alpha=0.6)
        ax.plot(x_range.cpu(), y_linear.cpu(), ':', label=r'Linear $\omega \cdot x$', alpha=0.6)
        ax.plot(x_range.cpu(), y_final.cpu(), 'r', lw=2, label='Total Activation')
        ax.plot(x_range.cpu(), x_range.cpu(), 'k', alpha=0.1, label='Identity')
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.2)
        ax.minorticks_on()  # 开启次网格
        ax.set_title(f"L{layer_idx}-N{neuron_idx} Edge{edge_idx}: Activation Function", fontsize=14)
        ax.legend()
    plt.show()


def visualize_params_and_active_edges(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    plt.subplots_adjust(hspace=0.3)
    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item(),
        'Temp (T)': torch.abs(layer.temperature).item()}
    ax1.bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green', 'red'])
    ax1.set_title(f"L{layer_idx}-N{neuron_idx}: Training Parameters", fontsize=14)
    with torch.no_grad():
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
        tau = torch.abs(layer.tau)
        active_energy = w_strength.clone()
        active_energy[w_strength <= tau] = np.nan
        current_cmap = plt.cm.YlGn.copy()
        current_cmap.set_bad(color='white')
        sns.heatmap(active_energy.cpu().numpy(), ax=ax2, cmap=current_cmap,
                    annot=True, fmt=".2f", annot_kws={"size": 8}, cbar_kws={'label': 'Energy'})
        ax2.set_title("Active Edges (Annotated by Energy Value)", fontsize=14)
    plt.show()


def visualize_attention_waveforms_v2(raw_segment, processed_segment, labels_segment, col_ratio):
    from matplotlib.collections import LineCollection
    weights_expanded = np.repeat(col_ratio, 2)
    t = np.linspace(0, 0.5, 250)  
    event_names = ['HandStart', 'Grasp', 'LiftOff', 'Hover', 'Replace', 'BothRel']
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 0.8]})
    plt.subplots_adjust(hspace=0.2)
    for i, (wave, title) in enumerate([(raw_segment[:, 0], "Raw Waveform"),
                                       (processed_segment[0, :], "Processed Feature")]):
        points = np.array([t, wave]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(vmin=-weights_expanded.max() * 0.1, vmax=weights_expanded.max())
        lc = LineCollection(segments, cmap='Blues', norm=norm)
        lc.set_array(weights_expanded[:-1])
        lc.set_linewidth(2.5)
        axes[i].add_collection(lc)
        axes[i].set_xlim(t.min(), t.max())
        axes[i].set_ylim(wave.min() * 1.2, wave.max() * 1.2)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"Attention-Guided {title}")
        axes[i].grid(True, alpha=0.3, linestyle=':')

    for i in range(6):
        event_line = labels_segment[:, i] * 0.8 + i
        axes[2].step(t, event_line, where='post', color='royalblue', lw=2, alpha=0.9)
        axes[2].text(-0.02, i + 0.2, event_names[i], ha='right', va='center', fontsize=9)

    axes[2].set_yticks(range(6))
    axes[2].set_yticklabels([])  
    axes[2].set_ylim(-0.5, 6)
    axes[2].set_title("Ground Truth: Event Occurrences")
    axes[2].set_xlabel("Time (s)")
    axes[2].minorticks_on()
    axes[2].grid(which='major', axis='x', linestyle='-', alpha=0.3)
    axes[2].grid(which='minor', axis='x', linestyle=':', alpha=0.1)
    plt.show()


def visualize_w3_and_3d(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = weight_layer.in_features
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0, :].numpy()
    k_size = len(kernel)
    from scipy.linalg import toeplitz
    first_col = np.zeros(seq_len)
    first_col[:k_size // 2 + 1] = kernel[k_size // 2:]
    first_row = np.zeros(seq_len)
    first_row[:k_size // 2 + 1] = kernel[k_size // 2::-1]
    toe_mat = toeplitz(first_col, first_row)
    fig3d = plt.figure(figsize=(12, 10))
    ax3d = fig3d.add_subplot(111, projection='3d')
    if layer.last_W1 is not None:
        W1 = layer.last_W1[0].cpu().numpy()
        alpha, beta, theta = [torch.abs(getattr(weight_layer, p)).item() for p in ['alpha', 'beta', 'theta']]
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)
        W_smooth = gaussian_filter(zoom(W_combined, 2, order=3), sigma=0.5)
        X, Y = np.meshgrid(np.linspace(0, seq_len - 1, W_smooth.shape[0]),
                           np.linspace(0, seq_len - 1, W_smooth.shape[0]))
        surf = ax3d.plot_surface(X, Y, W_smooth, cmap=plt.cm.Blues, alpha=0.95, rcount=200, ccount=200)
        fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
        ax3d.set_title(f"L{layer_idx}-N{neuron_idx}: Full 3D Fusion Matrix")
        ax3d.view_init(elev=30, azim=250)
    plt.show()


def visualize_2d_fusion_matrix(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = weight_layer.in_features
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0, :].numpy()
    k_size = len(kernel)
    from scipy.linalg import toeplitz
    first_col = np.zeros(seq_len)
    first_col[:k_size // 2 + 1] = kernel[k_size // 2:]
    first_row = np.zeros(seq_len)
    first_row[:k_size // 2 + 1] = kernel[k_size // 2::-1]
    toe_mat = toeplitz(first_col, first_row)
    if layer.last_W1 is not None:
        W1 = layer.last_W1[0].cpu().numpy()
        alpha = torch.abs(weight_layer.alpha).item()
        beta = torch.abs(weight_layer.beta).item()
        theta = torch.abs(weight_layer.theta).item()
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)
        plt.figure(figsize=(10, 8))
        sns.heatmap(W_combined, cmap=plt.cm.Blues, cbar_kws={'label': 'Combined Weight Strength'})
        plt.title(f"L{layer_idx}-N{neuron_idx}: 2D Fusion Matrix (β*W1 + θ*W3 + α*I)", fontsize=14)
        plt.xlabel("Pixel/Feature Index")
        plt.ylabel("Query Index")
        plt.grid(True, which='both', linestyle=':', color='white', alpha=0.2)
        plt.show()


def visualize_network_topology(model):
    model.eval()
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, FastKANLayer):
            layers.append(m)
    if not layers:
        print(" FastKANLayer is not ")
        return
    layer_sizes = [layers[0].in_neurons] + [l.out_neurons for l in layers]
    num_layers = len(layer_sizes)
    fig, ax = plt.subplots(figsize=(16, 10))
    pos = {}
    x_space = 2.0  
    for l_idx, size in enumerate(layer_sizes):
        x = l_idx * x_space
        y_positions = np.linspace(-size / 2, size / 2, size)
        for n_idx, y in enumerate(y_positions):
            pos[(l_idx, n_idx)] = (x, y)
    all_lines = []
    colors = []
    max_energy = 0.0
    
    with torch.no_grad():
        for l_idx, layer in enumerate(layers):
            w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
            tau = torch.abs(layer.tau)
            in_dim, out_dim = w_strength.shape
            for i in range(in_dim):
                for j in range(out_dim):
                    energy = w_strength[i, j].item()
                    threshold = tau[i, j].item()
                    if energy > threshold:
                        start_pos = pos[(l_idx, i)]
                        end_pos = pos[(l_idx + 1, j)]
                        all_lines.append([start_pos, end_pos])
                        colors.append(energy)
                        max_energy = max(max_energy, energy)

    widths = 1.0 + 3.0 * (np.array(colors) / (max(colors) + 1e-8))
    lc = LineCollection(all_lines, cmap=plt.cm.Blues, array=np.array(colors),
                        linewidths=widths, alpha=0.8)
    line_obj = ax.add_collection(lc)
    for (l_idx, n_idx), (x, y) in pos.items():
        ax.scatter(x, y, s=150, c='white', edgecolors='black', zorder=3)
    ax.set_title("BCI-KAN Global Topology: Active Paths Audit", fontsize=16)
    ax.set_xlabel("Processing Stage (Layers)", fontsize=12)
    plt.xticks(np.arange(num_layers) * x_space,
               ['Input'] + [f'Layer {i + 1}' for i in range(num_layers - 1)])
    ax.get_yaxis().set_visible(False)  
    cbar = fig.colorbar(line_obj, ax=ax)
    cbar.set_label(r'Edge Energy ($E = mean|W| + |\omega|$)')
    plt.grid(False)
    plt.show()


def analyze_eeg_representation(model, dataloader, device, n_samples=2000):
    model.eval()
    all_features = []
    all_labels = []
    collected = 0
    with torch.no_grad():
        for batch_data, batch_label in dataloader:
            latent = model(batch_data.to(device), return_latent=True)
            flat_feat = latent.reshape(latent.size(0), -1)
            all_features.append(flat_feat.cpu().numpy())
            label_summary = torch.argmax(batch_label, dim=1)  
            all_labels.append(label_summary.numpy())
            collected += batch_data.size(0)
            if collected >= n_samples: break
    X = np.concatenate(all_features, axis=0)
    y_summary = np.concatenate(all_labels, axis=0)
    X_scaled = StandardScaler().fit_transform(X)
    print(" UMAP and HDBSCAN are under processing ")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=25)
    cluster_labels = clusterer.fit_predict(embedding)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    event_names = ['HandStart', 'Grasp', 'LiftOff', 'Hover', 'Replace', 'BothRel']
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_summary,
                           cmap='tab10', s=10, alpha=0.5)
    ax1.set_title("Latent Space: Primary Event Labels", fontsize=14)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_ticks(range(6))
    cbar1.set_ticklabels(event_names)
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels,
                           cmap='Spectral', s=10, alpha=0.5)
    ax2.set_title("Latent Space: HDBSCAN Event Clusters", fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
    plt.suptitle("EEG Multi-Event Representation Analysis (N=6, L=125)", fontsize=16)
    plt.show()


def print_layer_parameter_summary(model):
    header = f"{'Layer Name':<12} | {'beta (W1)':>10} | {'theta (W3)':>10} | {'alpha (I)':>10} | {'gamma (Res)':>10} | {'omiga (Edge)':>12} | {'Temp (T)':>10}"
    print(header)
    print("-" * len(header))
    found_any = False
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight_layers') and hasattr(layer, 'omiga') and hasattr(layer, 'temperature'):
            found_any = True
            temp_val = torch.abs(layer.temperature).item()
            omiga_mean = torch.mean(torch.abs(layer.omiga)).item()
            betas, thetas, alphas, gammas = [], [], [], []
            for swl in layer.weight_layers:
                betas.append(torch.abs(swl.beta).item())   # Attention
                thetas.append(torch.abs(swl.theta).item()) # Convolution
                alphas.append(torch.abs(swl.alpha).item()) # Identity
                gammas.append(torch.abs(swl.gamma).item()) # Residual
            avg_beta = np.mean(betas)
            avg_theta = np.mean(thetas)
            avg_alpha = np.mean(alphas)
            avg_gamma = np.mean(gammas)
            print(f"{name:<12} | {avg_beta:10.4f} | {avg_theta:10.4f} | {avg_alpha:10.4f} | {avg_gamma:10.4f} | {omiga_mean:12.4f} | {temp_val:10.4f}")
    if not found_any:
        print("!!! Warning: No FastKANLayer found. Check model structure.")
import torch


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==================================================")
    print(f"Model Summary:")
    # print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"==================================================")
    return total_params, trainable_params


def calculate_model_flops(model, device):
    model.eval()
    dummy_input = torch.randn(1, 160, 250).to(device)
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        readable_flops, _ = clever_format([flops, 0], "%.3f")
        print(f"\n" + "=" * 50)
        print(f"   amount of FLOPs:  {readable_flops}")
        print(f"   GFLOPs : {flops / 1e9:.4f} G")
        print(f"=" * 50 + "\n")
        return flops
    except Exception as e:
        print(f"error: {e}")
        return 0


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = r"parameter.pth"
    data_root = "E:\\Python\\PythonProject\\dataset\\grasp-and-lift-eeg-detection\\train\\train"
    model = BCIKANModel().to(device)

    target_sub = 10 # which subject: (1, 12)
    target_ser = 7 # which movement: (1, 8)
    target_layer_idx = 5 # which layer: (0, 5)
    target_neuron_idx = 7 # which neuron: (0, 7)
    edge = (0, 1)   # which edge: (0:11, 0:7)
    raw_data_path = os.path.join(data_root, f'subj{target_sub}_series{target_ser}_data.csv')

    try:
        val_ds_example = WAYEEGDataset(
            data_root, subject_ids=[target_sub], series_ids=[target_ser],
            window_size=250, stride=250, is_train=True)
        valid_indices = []
        for idx in range(0, len(val_ds_example), 10): 
            _, last_label = val_ds_example[idx]
            if torch.any(last_label > 0.5): 
                valid_indices.append(idx)

        if valid_indices:
            target_idx = random.choice(valid_indices)
        else:
            target_idx = random.randint(0, len(val_ds_example) - 1)

        real_input, _ = val_ds_example[target_idx]
        start_phys_idx = val_ds_example.indices[target_idx]

        real_labels_segment = val_ds_example.raw_labels[start_phys_idx: start_phys_idx + 250]
        raw_df_example = pd.read_csv(raw_data_path).iloc[start_phys_idx: start_phys_idx + 250, 1:].values.astype(np.float32)
        real_input_tensor = real_input.unsqueeze(0).to(device, dtype=torch.float)

    except Exception as e:
        print(f"error: {e}")
        real_input_tensor = torch.randn(1, 160, 250).to(device)
        real_input = real_input_tensor[0].cpu()  
        raw_df_example = np.random.randn(250, 32)
        real_labels_segment = None

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

    target_layer = getattr(model, f'layer{target_layer_idx}')
    target_layer.visualize_idx = target_neuron_idx

    # calculate_model_flops(model, device)
    # count_parameters(model)
    # print_layer_parameter_summary(model)

    model.eval()
    with torch.no_grad():
        _ = model(real_input_tensor)
    audit_loader = WAYEEGDataset(
        data_root, subject_ids=[target_sub], series_ids=[target_ser],
        window_size=250, stride=125, is_train=True)
    audit_dataloader = torch.utils.data.DataLoader(audit_loader, batch_size=64, shuffle=True)
    
    # visualization
    analyze_eeg_representation(model, audit_dataloader, device, n_samples=3000)

    col_ratio = visualize_w1_analysis(target_layer, target_layer_idx, target_neuron_idx)

    visualize_attention_waveforms_v2(
        raw_df_example,
        real_input.numpy(),
        real_labels_segment,
        col_ratio)

    visualize_waveforms(raw_df_example, real_input.numpy())

    #visualize_spline_mapping_window(target_layer, target_layer_idx, target_neuron_idx, edge_idx=edge)
    #visualize_params_and_active_edges(target_layer, target_layer_idx, target_neuron_idx)

    visualize_w3_and_3d(target_layer, target_layer_idx, target_neuron_idx)

    visualize_2d_fusion_matrix(target_layer, target_layer_idx, target_neuron_idx)

    # visualize_network_topology(model)
    
