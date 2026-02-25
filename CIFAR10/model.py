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
from thop import profile，clever_format
from sklearn.preprocessing import StandardScaler
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter, zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataset_cifar10 import get_dataloader, add_coordinate_encoding


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


class SplineWeightLayer(nn.Module):   # neuron cell body
    def __init__(self, in_features=256, in_hidden=64, out_hidden=64, kernel_size=3):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  # 256 (16x16)
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # Identity
        self.beta = nn.Parameter(torch.ones(1))  # Attn
        self.gamma = nn.Parameter(torch.tensor([0.5]))  # Toeplitz
        self.theta = nn.Parameter(torch.ones(1))  # Residual
        self.W3_conv = nn.Conv2d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False)
        if in_hidden != out_hidden:
            self.residual_proj = nn.Linear(in_hidden, out_hidden)
        else:
            self.residual_proj = nn.Identity()
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x, W1=None):
        # x: [B, 256, Hidden]
        b, seq_len, h_dim = x.shape
        x_norm = self.ln(x)
        # x' = LN(x)*W2 + b
        x_prime = torch.matmul(x_norm, self.W2) + self.b  # [B, 256, out_hidden]
        x_spatial = x_prime.transpose(1, 2).view(b, self.out_hidden, 16, 16)
        w3_feat = self.W3_conv(x_spatial) 
        w3_feat = w3_feat.view(b, self.out_hidden, seq_len).transpose(1, 2)
        # (β*W1 + α*I) * x' + θ*W3(x')
        identity = torch.eye(seq_len, device=x.device).unsqueeze(0)
        if W1 is not None:
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat
        # add γ*res(x)
        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):   # edge
    def __init__(self, in_neurons, out_neurons, grid_size=64, spline_order=3, hidden_dim=64, neuron_out_dim=64, kernel_size=3):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)
        self.base_scales = nn.Parameter(torch.empty(in_neurons, out_neurons))
        nn.init.normal_(self.base_scales, mean=0.6, std=0.1)
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size) for _ in range(out_neurons)
        ])
        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.last_W1 = None
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))

    def b_spline_cubic(self, x):
        h = 2.0 / (self.num_coeffs - 1)
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res = torch.zeros_like(dist)
        mask1 = dist < 1.0
        res[mask1] = (2 / 3) - (dist[mask1] ** 2) + (dist[mask1] ** 3 / 2)
        mask2 = (dist >= 1.0) & (dist < 2.0)
        res[mask2] = ((2 - dist[mask2]) ** 3) / 6
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape
        x_energy_in = torch.mean(x_in, dim=-1)
        x_norm = torch.clamp(x_energy_in / (torch.abs(x_energy_in).max(dim=-1, keepdim=True)[0] + 1e-8), -0.99, 0.99)
        base_feat = F.silu(x_norm).unsqueeze(2) * self.base_scales.unsqueeze(-1).unsqueeze(0)
        basis = self.b_spline_cubic(x_norm)
        spline_feat = torch.einsum('bifc,ijc->bijf', basis, self.spline_weights)
        edge_response = (base_feat + spline_feat).unsqueeze(-1)
        omiga_expanded = torch.abs(self.omiga).view(1, self.in_neurons, self.out_neurons, 1, 1)
        activated_signals = (edge_response + omiga_expanded) * x_in.unsqueeze(2)
        next_outputs, next_projections = [], []
        t_val = torch.abs(self.temperature) + 1e-4
        
        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(torch.abs(current_edges), dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((edge_energies - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)
            W1_j = None
            if proj_x_prev is not None:
                multiplier = (edge_energies / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, i, :, :] for i in range(mid, mid * 2)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)
            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None
                
            combined_input = torch.mean(current_edges * mask, dim=1)
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


class ImagePrepConv(nn.Module):
    def __init__(self, in_channels=5, out_channels=32):
        super(ImagePrepConv, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels))
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU())
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU())
        
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels))

        self.relu = nn.SiLU()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        #self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):   # x: [B, 5, 32, 32]
        identity = self.shortcut(x)  # [B, 64, 32, 32]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)  # [B, 64, 32, 32]
        out = self.post_conv(out)
        out += identity
        out = self.relu(out)
        out = self.pixel_unshuffle(out)
        b, c, h, w = out.shape
        out = out.flatten(2)  # [B, 64, 256]
        out = out.transpose(1, 2)  # [B, 256, 64]
        out = out.unsqueeze(1)  # [B, 1, 256, 64]
        return out


class ECGKANModel(nn.Module):   # main E-KAT model
    def __init__(self, grid_size=64, spline_order=3):
        super(ECGKANModel, self).__init__()
        self.prep_conv = ImagePrepConv(in_channels=5, out_channels=32)
        
        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order, hidden_dim=128, neuron_out_dim=64, kernel_size=7)
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order, hidden_dim=64, neuron_out_dim=64, kernel_size=5)
        self.layer3 = FastKANLayer(8, 16, grid_size, spline_order, hidden_dim=64, neuron_out_dim=128, kernel_size=5)
        self.layer4 = FastKANLayer(16, 16, grid_size, spline_order, hidden_dim=128, neuron_out_dim=128, kernel_size=3)
        self.layer5 = FastKANLayer(16, 20, grid_size, spline_order, hidden_dim=128, neuron_out_dim=128, kernel_size=3)
        self.layer6 = FastKANLayer(20, 16, grid_size, spline_order, hidden_dim=128, neuron_out_dim=16, kernel_size=3)
        self.layer7 = FastKANLayer(16, 10, grid_size, spline_order, hidden_dim=16, neuron_out_dim=1, kernel_size=1)

    def forward(self, x, return_latent=False, audit_layer=None):# x: [B, 5, 32, 32]
        x = self.prep_conv(x)  # [B, 1, 256, 64]
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]
        projections = [None]
        for i, layer in enumerate(layers):
            layer_idx = i + 1
            if audit_layer == layer_idx:
                return x
            x, proj = layer(x, proj_x_prev=projections[-1])
            projections.append(proj)
        if return_latent:
            return x.squeeze(-1)
        return torch.mean(x, dim=(-1, -2))

    def get_active_conn_info(self):
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.base_scales)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {"active": active_count, "total": m.in_neurons * m.out_neurons,
                                  "tau_mean": torch.abs(m.tau).mean().item(), 'ratio': active_count / (m.in_neurons * m.out_neurons)}
        return info


# visualization
def visualize_model_internals(model, layer_idx=1, neuron_idx=0, edge_idx=(0, 0)):
    model.eval()
    layer = getattr(model, f'layer{layer_idx}')
    weight_layer = layer.weight_layers[neuron_idx]
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    sns.heatmap(weight_layer.W2.detach().cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W2 Matrix (Feature Projection)")
    
    if layer.last_W1 is not None:
        sns.heatmap(layer.last_W1[0].detach().cpu().numpy(), ax=axes[0, 1], cmap='magma')
        axes[0, 1].set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention (Global Space)")
    else:
        axes[0, 1].text(0.5, 0.5, "W1 is None", ha='center')
        
    with torch.no_grad():
        conv_weight = weight_layer.W3_conv.weight.detach().cpu()
        d_out, _, k, _ = conv_weight.shape
        sparse_w3 = []
        for d in range(d_out):
            kernel_2d = conv_weight[d, 0]  # [k, k]
            row_list = []
            for r in range(k):
                row_list.append(kernel_2d[r]) 
                if r < k - 1:
                    row_list.append(torch.zeros(16 - k))  
            sparse_w3.append(torch.cat(row_list))

        w3_display = torch.stack(sparse_w3).numpy()
        sns.heatmap(w3_display, ax=axes[1, 0], cmap='coolwarm', center=0)
        axes[1, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W3 Sparse Matrix (Local Prior)")
        axes[1, 0].set_xlabel("Sparse Kernel Index (3k + 2*(16-k))")
        axes[1, 0].set_ylabel("Channel Dimension (d)")

    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item()}
    axes[1, 1].bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green'])
    axes[1, 1].set_title(f"L{layer_idx}-N{neuron_idx}: Component Weights")
    axes[1, 1].set_ylim(0, max(params.values()) * 1.2)

    with torch.no_grad():
        x_range = torch.linspace(-1, 1, 200).to(layer.spline_weights.device)
        base_scale = layer.base_scales[edge_idx[0], edge_idx[1]]
        y_base = base_scale * F.silu(x_range)

        spl_func = layer.b_spline_cubic if hasattr(layer, 'b_spline_cubic') else layer.b_spl_cubic
        basis = spl_func(x_range.unsqueeze(0))
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        y_spline = torch.matmul(basis, coeffs).squeeze()

        axes[2, 0].plot(x_range.cpu(), y_base.cpu(), '--', label='Base (SiLU)', alpha=0.6)
        axes[2, 0].plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline', alpha=0.6)
        axes[2, 0].plot(x_range.cpu(), (y_base + y_spline).cpu(), 'r', lw=2, label='Combined')
        axes[2, 0].set_title(f"Activation: Edge ({edge_idx[0]} -> {edge_idx[1]})")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

    with torch.no_grad():
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.base_scales)
        active_mask = (w_strength > torch.abs(layer.tau)).float()
        sns.heatmap(active_mask.cpu().numpy(), ax=axes[2, 1], annot=False, cmap='Blues', cbar=False)
        axes[2, 1].set_title(f"Layer {layer_idx}: Edge Conductivity")
        axes[2, 1].set_xlabel("Output Neurons")
        axes[2, 1].set_ylabel("Input Neurons")
    plt.show()


def visualize_w1_analysis(layer, layer_idx, neuron_idx):
    if layer.last_W1 is None: return None
    W1_data = layer.last_W1[0].detach().cpu().numpy()
    col_ratio = np.sum(W1_data, axis=0) / (np.sum(W1_data) + 1e-8)
    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[25, 1], hspace=0.05, wspace=0.02)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(range(len(col_ratio)), col_ratio, color='skyblue', width=0.8)
    ax_bar.set_xlim(-0.5, len(col_ratio) - 0.5)
    ax_bar.set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention Analysis (Image Pixels)", fontsize=14)
    ax_bar.grid(True, linestyle='--', alpha=0.5)
    ax_heat = fig.add_subplot(gs[1, 0])
    sns.heatmap(W1_data, ax=ax_heat, cmap='magma', cbar=False)
    ax_heat.set_xlabel("Pixel Sequence Index (0-255)")
    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(W1_data.min(), W1_data.max()), cmap='magma'), cax=ax_cbar)
    plt.show()
    return col_ratio


def visualize_images_comparison(raw_img, label_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    img_show = np.clip(raw_img * std + mean, 0, 1).transpose(1, 2, 0)
    ax.imshow(img_show)
    ax.set_title(f"Initial Image: {label_name} (32x32 RGB)", fontsize=14)
    ax.axis('off') 
    plt.show()


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

        ax.plot(x_range.cpu(), y_spline.cpu(), color='#1f77b4', linestyle='--', label='Spline $f(x)$')
        ax.plot(x_range.cpu(), y_linear.cpu(), color='#ff7f0e', linestyle=':', label=r'Linear $\omega \cdot x$')
        ax.plot(x_range.cpu(), y_final.cpu(), 'r', lw=2.5, label='Total Activation')
        ax.plot(x_range.cpu(), x_range.cpu(), color='lightgray', alpha=0.5, label='Identity')

        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.6', color='lightgray', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray', alpha=0.3)
        ax.set_title(f"L{layer_idx}-N{neuron_idx} Edge{edge_idx}: Activation Function")
        ax.legend()
    plt.show()


def visualize_params_and_active_edges(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    plt.subplots_adjust(hspace=0.3)

    params = {'alpha': torch.abs(weight_layer.alpha).detach().item(),
              'beta': torch.abs(weight_layer.beta).detach().item(),
              'theta': torch.abs(weight_layer.theta).detach().item(),
              'gamma': torch.abs(weight_layer.gamma).detach().item()}
    ax1.bar(params.keys(), params.values(), color='royalblue', alpha=0.7)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
    active_energy = w_strength.clone()
    active_energy[w_strength <= torch.abs(layer.tau)] = np.nan
    sns.heatmap(active_energy.detach().cpu().numpy(), ax=ax2, cmap='YlGn', annot=True, fmt=".2f", linewidths=1)
    plt.show()


def visualize_attention_images(raw_img, col_ratio, label_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={'width_ratios': [1, 1.07]})
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    rgb_img = np.clip(raw_img * std + mean, 0, 1).transpose(1, 2, 0)
    attn_mask = col_ratio.reshape(16, 16)
    attn_mask_norm = (attn_mask - attn_mask.min()) / (attn_mask.max() - attn_mask.min() + 1e-8)
    attn_upsampled = zoom(attn_mask_norm, 2, order=1)
    
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Initial RGB Image: {label_name}", fontsize=12)
    axes[0].axis('off')
    bw_img = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
    axes[1].imshow(bw_img, cmap='gray')

    im = axes[1].imshow(attn_upsampled, cmap='Reds', alpha=0.7)
    axes[1].set_title("Attention Heatmap (Overlaid)", fontsize=12)
    axes[1].axis('off')

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 颜色条放置在侧边且不占主图位
    plt.colorbar(im, cax=cax, label='Attention Intensity')
    plt.suptitle(f"Audit Analysis: Category [{label_name}]", fontsize=16, y=0.95)
    plt.show()


def visualize_fusion_matrices(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = 256  # 16x16
    H, W = 16, 16
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0].numpy()
    k_size = kernel.shape[0]
    pad = k_size // 2

    toe_mat = np.zeros((seq_len, seq_len))
    for i in range(H):
        for j in range(W):
            row_idx = i * W + j
            for m in range(-pad, pad + 1):
                for n in range(-pad, pad + 1):
                    ni, nj = i + m, j + n
                    if 0 <= ni < H and 0 <= nj < W:
                        col_idx = ni * W + nj
                        toe_mat[row_idx, col_idx] = kernel[m + pad, n + pad]

    if layer.last_W1 is not None:
        W1 = layer.last_W1[0].cpu().numpy()
        alpha, beta, theta = [torch.abs(getattr(weight_layer, p)).item() for p in ['alpha', 'beta', 'theta']]
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)
        fig3d = plt.figure(figsize=(12, 10))
        ax3d = fig3d.add_subplot(111, projection='3d')
        W_smooth = gaussian_filter(W_combined, sigma=0.4)
        X, Y = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
        ax3d.plot_surface(X, Y, W_smooth, cmap=plt.cm.Blues, alpha=0.8, rcount=128, ccount=128)
        ax3d.view_init(elev=30, azim=250)
        plt.show()
        plt.figure(figsize=(10, 8))
        sns.heatmap(W_combined, cmap=plt.cm.Blues)
        plt.title(f"L{layer_idx}-N{neuron_idx}: 2D Fusion Matrix (Block Toeplitz)")
        plt.show()


def visualize_network_topology(model):
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, FastKANLayer)]
    layer_sizes = [layers[0].in_neurons] + [l.out_neurons for l in layers]
    fig, ax = plt.subplots(figsize=(16, 10))
    pos = {}
    x_space = 2.0
    for l_idx, size in enumerate(layer_sizes):
        x = l_idx * x_space
        y_positions = np.linspace(-size / 2, size / 2, size)
        for n_idx, y in enumerate(y_positions):
            pos[(l_idx, n_idx)] = (x, y)

    all_lines, colors = [], []
    with torch.no_grad():
        for l_idx, layer in enumerate(layers):
            w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
            tau = torch.abs(layer.tau)
            for i in range(w_strength.shape[0]):
                for j in range(w_strength.shape[1]):
                    if w_strength[i, j] > tau[i, j]:
                        all_lines.append([pos[(l_idx, i)], pos[(l_idx + 1, j)]])
                        colors.append(w_strength[i, j].item())

    from matplotlib.collections import LineCollection
    lc = LineCollection(all_lines, cmap=plt.cm.Blues, array=np.array(colors),
                        linewidths=1.0 + 3.0 * (np.array(colors) / (max(colors) + 1e-8)), alpha=0.8)
    ax.add_collection(lc)
    for p in pos.values(): ax.scatter(p[0], p[1], s=120, c='white', edgecolors='black', zorder=3)
    ax.set_title("CIFAR-KAN Global Topology")
    plt.show()


def analyze_image_representation(model, dataloader, device, n_samples=3000):
    model.eval()
    all_features = []
    all_labels = []
    collected = 0
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        for batch_data, batch_label in dataloader:
            input_5ch = add_coordinate_encoding(batch_data.to(device))
            latent = model(input_5ch, return_latent=True)
            flat_feat = latent.reshape(latent.size(0), -1)  # [B, 2560]
            all_features.append(flat_feat.cpu().numpy())
            all_labels.append(batch_label.numpy())
            collected += batch_data.size(0)
            if collected >= n_samples: break
    X = np.concatenate(all_features, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=40)
    cluster_labels = clusterer.fit_predict(embedding)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_true,
                           cmap='tab10', s=15, alpha=0.6)
    ax1.set_title("CIFAR-10 Latent Space: Ground Truth (10 Classes)", fontsize=14)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_ticks(range(10))
    cbar1.set_ticklabels(cifar10_classes)

    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels,
                           cmap='Spectral', s=15, alpha=0.6)
    ax2.set_title("CIFAR-10 Latent Space: HDBSCAN Clusters", fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID (-1: Noise)')
    plt.suptitle("CIFAR-10 Output Layer Feature Audit (Flattened 2560D Representation)", fontsize=16)
    plt.show()


def visualize_convolution_residual_final(model, layer_idx, neuron_idx, x_prime_input, sample_img, device):
    layer = getattr(model, f'layer{layer_idx}')
    weight_layer = layer.weight_layers[neuron_idx]
    model.eval()
    with torch.no_grad():
        if x_prime_input.dim() == 4:
            x_target = x_prime_input.squeeze(1) if x_prime_input.size(1) == 1 else x_prime_input[:, 0, :, :]
        else:
            x_target = x_prime_input
        x_norm = weight_layer.ln(x_target)
        x_projected = torch.matmul(x_norm, weight_layer.W2) + weight_layer.b
        b, seq_len, h_dim = x_projected.shape
        side = int(np.sqrt(seq_len))
        x_spatial = x_projected.transpose(1, 2).view(b, h_dim, side, side)
        w3_output = weight_layer.W3_conv(x_spatial)
        delta_x = torch.abs(weight_layer.theta) * w3_output
        delta_energy = torch.mean(torch.abs(delta_x[0]), dim=0).cpu().numpy()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(delta_energy, cmap='magma')
    plt.title(r"Layer {} Conv Residual: $\Delta x = \theta \cdot W3$".format(layer_idx), fontsize=12)
    plt.colorbar(im, label=r"Residual Intensity ($\theta \cdot W3$)")
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


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==================================================")
    print(f"Model Summary:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"==================================================")
    return total_params, trainable_params


def calculate_model_complexity(model, device):
    model.eval()
    dummy_input = torch.randn(1, 5, 32, 32).to(device)
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        readable_flops, readable_params = clever_format([flops, params], "%.3f")
        print(f"\n" + "=" * 50)
        print(f"   amount of FLOPs:  {readable_flops}")
        print(f"   (GFLOPs):    {flops / 1e9:.4f} G")
        print(f"=" * 50 + "\n")
        return flops, params
    except Exception as e:
        print(f"error: {e}")
        return 0, 0


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGKANModel(grid_size=64, spline_order=3).to(device)
    train_loader, test_loader, _ = get_dataloader(batch_size=1)

    checkpoint_path = 'parameter.pth'
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    #analyze_image_representation(model, audit_loader, device, n_samples=3000)

    target_sample_idx = 1   # sample index
    target_layer_idx, target_neuron_idx = 5, 10  # which layer which neuron (0:6, 0:15)
    edge = (5, 5)   # which edge

    iterator = iter(test_loader)
    for _ in range(target_sample_idx + 1):
        sample_img, sample_label = next(iterator)

    target_layer = getattr(model, f'layer{target_layer_idx}')
    target_layer.visualize_idx = target_neuron_idx
    label_str = cifar10_classes[sample_label.item()]
    # calculate_model_complexity(model, device)

    model.eval()
    with torch.no_grad():
        input_5ch = add_coordinate_encoding(sample_img.to(device))
        # x_prime_input = model(input_5ch, audit_layer=target_layer_idx)
        logits = model(input_5ch)
        probs = F.softmax(logits, dim=1) 
        conf, pred_idx = torch.max(probs, dim=1)

    true_idx = sample_label.item()
    predicted_idx = pred_idx.item()
    true_name = cifar10_classes[true_idx]
    pred_name = cifar10_classes[predicted_idx]
    is_correct = true_idx == predicted_idx
    status_str = "[CORRECT] ✓" if is_correct else "[WRONG] ✗"

    print(f"\n" + "=" * 50)
    print(f"real label: [{true_name}] | predicted label: [{pred_name}]")
    print(f"T or F: {status_str}")
    print(f"Prediction confidence level: {conf.item() * 100:.2f}%")  

    # visualization

    # visualize_convolution_residual_final(model, target_layer_idx, target_neuron_idx, x_prime_input, sample_img, device)
    # col_ratio = visualize_w1_analysis(target_layer, target_layer_idx, target_neuron_idx)
    # visualize_attention_images(sample_img[0].numpy(), col_ratio, label_str)
    # # visualize_images_comparison(sample_img[0].numpy(), label_str)
    # visualize_spline_mapping_window(target_layer, target_layer_idx, target_neuron_idx, edge)
    # # visualize_params_and_active_edges(target_layer, target_layer_idx, target_neuron_idx)
    # visualize_fusion_matrices(target_layer, target_layer_idx, target_neuron_idx)
    # #visualize_network_topology(model)

