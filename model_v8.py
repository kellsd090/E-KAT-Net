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
from sklearn.preprocessing import StandardScaler
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.collections import LineCollection
from database_mit import prepare_ecg_data


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


class SplineWeightLayer(nn.Module):
    def __init__(self, in_features=187, in_hidden=64, out_hidden=64, kernel_size=15):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  # 序列长度 750
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size

        # 1. 特征投影与独立位置偏置
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        # 保持每个采样点独立的偏置参数
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)

        # 2. 权重控制系数
        self.alpha = nn.Parameter(torch.ones(1))  # I 的权重
        self.beta = nn.Parameter(torch.ones(1))  # W1 的权重
        self.theta = nn.Parameter(torch.ones(1))  # W3 的权重
        self.gamma = nn.Parameter(torch.ones(1))  # 残差权重

        # --- W3 1D 时序卷积先验 ---
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

        # W3 路径
        x_temporal = x_prime.transpose(1, 2)
        w3_feat = self.W3_conv(x_temporal).transpose(1, 2)

        # W1 路径 (基于 Requirement 2 的自注意力输入)
        if W1 is not None:
            # 使用预存的 identity
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * self.identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat

        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, grid_size=64, spline_order=3, hidden_dim=64, neuron_out_dim=64,
                 kernel_size=15, seq_len=187):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order

        # 纯粹的 B-样条权重，去掉了 base_scales
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)

        # 每一个输出神经元对应一个 1D 优化的 SplineWeightLayer
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_features=seq_len, in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size)
            for _ in range(out_neurons)
        ])

        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))
        self.last_W1 = None

        # 将原本的随机初始化改为线性初始化
        # 目标：让输出 y = Spline(x) 在初始时接近 y = x
        with torch.no_grad():
            # 生成从 -1.0 到 1.0 的等差序列，代表 y=x 的趋势
            lin_init = torch.linspace(-1.0, 1.0, self.num_coeffs).to(self.spline_weights.device)
            # 形状调整为 [in_n, out_n, num_coeffs]
            initial_weights = lin_init.view(1, 1, -1).repeat(self.in_neurons, self.out_neurons, 1)
            # 稍微加一点极小的噪声打破对称性
            self.spline_weights.data = initial_weights + torch.randn_like(initial_weights) * 0.01

    def b_spline_cubic(self, x):
        """
        优化版三次 B-样条：通过数学公式合并分支，消除 mask 赋值
        """
        h = 2.0 / (self.num_coeffs - 1)
        # [B, in_n, S, D, 1] - [num_coeffs] -> [B, in_n, S, D, num_coeffs]
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h

        # 使用 ReLU 截断代替 mask，确保 dist < 2.0 之外的部分为 0
        # 通过组合多项式模拟原有的分段逻辑
        # 核心区 (dist < 1) 和 边缘区 (1 <= dist < 2)

        # 逻辑 1: 计算 (2 - dist)^3 / 6 (覆盖 0 <= dist < 2)
        res_outer = torch.relu(2.0 - dist) ** 3 / 6.0

        # 逻辑 2: 减去核心区多算的补偿值 (只针对 dist < 1)
        # 原逻辑核心区为: 2/3 - dist^2 + dist^3/2
        # 经过代数化简，补偿项为: 4/6 * (1 - dist)^3
        res_inner = torch.relu(1.0 - dist) ** 3 * (4.0 / 6.0)

        # 最终结果 = 外部大包络 - 内部补偿补偿
        res = res_outer - res_inner
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape # x_in: [B, in_n, 200, 128]

        # 1. 样条输入归一化
        max_val = torch.max(torch.abs(x_in)) + 1e-8
        x_norm = (x_in / max_val) * 0.95
        x_norm = torch.clamp(x_norm, -0.99, 0.99)

        # 2. Requirement 1: 纯 B-样条点乘逻辑
        basis = self.b_spline_cubic(x_norm)
        spline_mapping = torch.einsum('binfc,ioc->bio nf', basis, self.spline_weights)
        # 2. 加上线性残差项 omega * x
        # omega 现在扮演的是“线性增益”或“残差系数”的角色
        omega_expanded = torch.abs(self.omiga).view(1, in_n, self.out_neurons, 1, 1)
        activated_signals = spline_mapping + omega_expanded * x_in.unsqueeze(2)

        next_outputs, next_projections = [], []
        # 针对 750 长度调整温度系数，防止梯度饱和
        t_val = torch.abs(self.temperature) * np.sqrt(seq_len / 256.0) + 1e-4

        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(current_edges ** 2, dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((torch.sqrt(edge_energies + 1e-8) - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)

            # 3. Requirement 2: 保持自注意力逻辑
            W1_j = None
            if proj_x_prev is not None:
                multiplier = (torch.sqrt(edge_energies + 1e-8) / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask

                # 拼接所有输入神经簇的特征进行自注意力
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, 2*i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, 2*i+1, :, :] for i in range(mid)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            # 融合并输入神经元核心
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
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScalePrepConv(nn.Module):
    def __init__(self):
        super(MultiScalePrepConv, self).__init__()
        # 三个并行分支，捕捉不同频率特征
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), # 抓高频细节
            nn.BatchNorm1d(16),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=15, padding=7), # 抓标准形态
            nn.BatchNorm1d(40),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=25, padding=12), # 抓长程节律(R-R间期)
            nn.BatchNorm1d(8),
            nn.SiLU()
        )
        # self.se = SEBlock(32) # 对 8+16+8=32 通道进行统一权重重分配

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # 拼接维度：[B, 32, 187]
        out = torch.cat([x1, x2, x3], dim=1)
        return out # self.se(out)


class ECGKANModel(nn.Module):
    def __init__(self, grid_size=64, spline_order=3, seq_len=187):
        super(ECGKANModel, self).__init__()
        self.prep_conv = MultiScalePrepConv()
        # Layer 1 & 2: 保持 32 维特征
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

        if return_latent:
            # 返回原始特征张量 [B, 5, 187]
            return x.squeeze(-1)

        # 默认返回分类结果 [B, 5]
        return torch.mean(x, dim=(-1, -2))

    def get_active_conn_info(self):
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    # 核心修正：将 m.base_scales 替换为 m.omiga
                    # 在超集模型中，omiga 代表线性分支的权重强度
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.omiga)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()

                    info[name] = {
                        "active": active_count,
                        "total": m.in_neurons * m.out_neurons,
                        "tau_mean": torch.abs(m.tau).mean().item(),
                        "ratio": active_count / (m.in_neurons * m.out_neurons + 1e-8)
                    }
        return info


def visualize_model_internals(model, layer_idx=1, neuron_idx=0, edge_idx=(0, 0)):
    model.eval()
    layer = getattr(model, f'layer{layer_idx}')
    # 确保兼容性：如果 visualize_model_internals 里的 layer.b_spl_cubic 指向正确的 b_spline_cubic
    if not hasattr(layer, 'b_spl_cubic'):
        layer.b_spl_cubic = layer.b_spline_cubic

    weight_layer = layer.weight_layers[neuron_idx]

    # 布局：3行2列
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # --- (1) W2 矩阵热力图 ---
    sns.heatmap(weight_layer.W2.detach().cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W2 Matrix (Feature Projection)")

    # --- (2) W1 矩阵热力图 (动态生成) ---
    if layer.last_W1 is not None:
        sns.heatmap(layer.last_W1[0].detach().cpu().numpy(), ax=axes[0, 1], cmap='magma')
        axes[0, 1].set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention (Global Space)")
    else:
        axes[0, 1].text(0.5, 0.5, "W1 is None", ha='center')

    # --- (3) [修正] W3 矩阵可视化 (适配 1D 卷积) ---
    with torch.no_grad():
        # 提取 1D 卷积核权重: [out_hidden, 1, kernel_size]
        conv_weight = weight_layer.W3_conv.weight.detach().cpu()
        d_out, _, k = conv_weight.shape
        # 直接展示 1D 核的热力图，纵轴是通道，横轴是时间窗口
        sns.heatmap(conv_weight.squeeze(1).numpy(), ax=axes[1, 0], cmap='coolwarm', center=0)
        axes[1, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W3 1D Convolutional Kernels")
        axes[1, 0].set_xlabel("Time Window (k)")
        axes[1, 0].set_ylabel("Hidden Channels")

    # --- (4) θ, α, β, γ 参数柱状图 ---
    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item()
    }
    axes[1, 1].bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green'])
    axes[1, 1].set_title(f"L{layer_idx}-N{neuron_idx}: Component Weights")

    # --- (5) [修正] B-样条激活映射函数可视化 ---
    with torch.no_grad():
        # 定义测试量程：从 -1 到 1 (对应我们归一化后的 x_norm)
        x_range = torch.linspace(-1, 1, 200).to(layer.spline_weights.device)

        # 获取当前边的 omiga (线性残差系数)
        current_omiga = torch.abs(layer.omiga[edge_idx[0], edge_idx[1]])

        # 1. 计算纯样条映射输出: f(x)
        # basis 形状: [1, 200, num_coeffs]
        basis = layer.b_spline_cubic(x_range.unsqueeze(0))
        # coeffs 形状: [num_coeffs]
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        # y_spline 形状: [200]
        y_spline = torch.matmul(basis, coeffs).squeeze()

        # 2. 计算线性捷径输出: omega * x
        y_linear = current_omiga * x_range

        # 3. 最终组合输出: y = f(x) + omega * x (替代了原来的 SiLU)
        y_final = y_spline + y_linear

        # 绘图
        axes[2, 0].plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline Mapping $f(x)$', alpha=0.7)
        axes[2, 0].plot(x_range.cpu(), y_linear.cpu(), ':', label=r'Linear Shortcut $\omega \cdot x$', alpha=0.7)
        axes[2, 0].plot(x_range.cpu(), y_final.cpu(), 'r', lw=2.5, label='Total Activation (Learned)')

        # 辅助线：绘制 y=x 虚线作为参考，看看模型偏离线性多远
        axes[2, 0].plot(x_range.cpu(), x_range.cpu(), 'k', alpha=0.2, lw=1, label='Identity (y=x)')

        axes[2, 0].set_title(f"Edge ({edge_idx[0]}->{edge_idx[1]}): Value Mapping Function")
        axes[2, 0].set_xlabel("Input Normalized Voltage")
        axes[2, 0].set_ylabel("Output Mapped Voltage")
        axes[2, 0].legend(loc='upper left', fontsize='small')
        axes[2, 0].grid(True, alpha=0.3)

    # --- (6) 层间导通热力图 (基于 omiga) ---
        # --- (6) 层间导通热力图 (基于 RMS 强度判定) ---
    with torch.no_grad():
        # 粗略估计每条边的“平均表现强度”
        # 样条权重的均值 + omiga 线性强度
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
        # 注意：这里的判断逻辑应与 forward 里的 mask 逻辑对应
        active_mask = (w_strength > torch.abs(layer.tau)).float()
        sns.heatmap(active_mask.cpu().numpy(), ax=axes[2, 1], cmap='Greens', cbar=False)

    plt.show()


# --- 1. W1 深度分析窗口 ---
def visualize_w1_analysis(layer, layer_idx, neuron_idx):
    if layer.last_W1 is None: return None
    W1_data = layer.last_W1[0].detach().cpu().numpy()
    col_ratio = np.sum(W1_data, axis=0) / (np.sum(W1_data) + 1e-8)

    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[25, 1], hspace=0.05, wspace=0.02)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(range(len(col_ratio)), col_ratio, color='skyblue', width=0.8)
    ax_bar.set_xlim(-0.5, len(col_ratio) - 0.5)
    ax_bar.set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention Analysis (Sparse Grid)", fontsize=14)
    ax_bar.grid(True, linestyle='--', alpha=0.5)

    ax_heat = fig.add_subplot(gs[1, 0])
    sns.heatmap(W1_data, ax=ax_heat, cmap='magma', cbar=False)
    ax_heat.set_xlabel("Time/Feature Index")

    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(W1_data.min(), W1_data.max()), cmap='magma'), cax=ax_cbar)
    plt.show()
    return col_ratio


# --- 2. 原始与预处理对比 (适配单通道 ECG) ---
def visualize_waveforms(raw_segment, processed_segment, sample_rate=125):
    t = np.linspace(0, len(raw_segment) / sample_rate, len(raw_segment))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(t, raw_segment, color='gray', lw=1.2)
    axes[0].set_title("ECG Raw Waveform (Normalized [-1, 1])")
    # 这里的 processed_segment 对应 MultiScalePrepConv 后的 64 通道特征
    axes[1].plot(t, processed_segment[0, :], color='black', lw=1.2)
    axes[1].set_title("ECG Processed Feature (First MS-Conv Branch)")
    for a in axes: a.grid(True, alpha=0.3, linestyle=':')
    plt.show()


# --- 3. B-样条映射图 ---
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
        ax.plot(x_range.cpu(), y_final.cpu(), 'r', lw=2, label='Total Activation')
        ax.plot(x_range.cpu(), x_range.cpu(), 'k', alpha=0.1, label='Identity')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.show()


# --- 4. 参数审计与能量标注 ---
def visualize_params_and_active_edges(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    params = {'alpha': torch.abs(weight_layer.alpha).item(), 'beta': torch.abs(weight_layer.beta).item(),
              'theta': torch.abs(weight_layer.theta).item(), 'gamma': torch.abs(weight_layer.gamma).item()}
    ax1.bar(params.keys(), params.values(), color='royalblue', alpha=0.7)
    w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
    active_energy = w_strength.clone()
    active_energy[w_strength <= torch.abs(layer.tau)] = np.nan
    sns.heatmap(active_energy.detach().cpu().numpy(), ax=ax2, cmap='YlGn', annot=True, fmt=".2f", linewidths=1)
    plt.show()


# --- 5. 注意力引导波形 (ECG 增强版) ---
def visualize_attention_waveforms_v2(raw_segment, processed_segment, col_ratio):
    """
    在一个窗口内绘制两张图：上方为原始波形，下方为预处理特征，均带有注意力着色。
    raw_segment: 原始 ECG 信号 [187]
    processed_segment: 预处理后的特征 [187]
    col_ratio: W1 层的注意力权重
    """
    from matplotlib.collections import LineCollection
    t = np.linspace(0, 1.0, len(raw_segment))

    # 1. 将注意力权重插值到信号长度
    weights_expanded = np.interp(np.linspace(0, 1, len(raw_segment)),
                                 np.linspace(0, 1, len(col_ratio)),
                                 col_ratio)

    # 2. 设置统一的颜色映射标准
    norm = plt.Normalize(vmin=-weights_expanded.max() * 0.1, vmax=weights_expanded.max())

    # 3. 创建 2 行 1 列的子图窗口
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)  # 调整上下间距

    titles = ["Raw ECG Waveform: Attention Shading", "Processed Feature: Attention Shading"]
    data_list = [raw_segment, processed_segment]
    colors = ['gray', 'black']

    for i, ax in enumerate(axes):
        wave_data = data_list[i]

        # 构造线段集合以便分段着色
        points = np.array([t, wave_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 创建 LineCollection 并设置 Blues 配色
        lc = LineCollection(segments, cmap='Blues', norm=norm, lw=2.5, zorder=3)
        lc.set_array(weights_expanded[:-1])

        # 绘制背景淡色参考线，确保低权重处波形依然可见
        ax.plot(t, wave_data, color=colors[i], alpha=0.1, lw=1, zorder=1)

        ax.add_collection(lc)
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(wave_data.min() * 1.5, wave_data.max() * 1.5)

        # 添加网格与标题
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(titles[i], fontsize=14)

        # 为每个子图添加颜色条
        plt.colorbar(lc, ax=ax, label='W1 Weight')

    axes[1].set_xlabel("Time (Normalized)")
    plt.show()


# --- 6. 3D 与 7. 2D 融合矩阵 ---
def visualize_fusion_matrices(layer, layer_idx, neuron_idx):
    """窗口 (6 & 7): 整合 3D 与 2D 融合矩阵"""
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = weight_layer.in_features

    # 提取 W3 卷积核并构造 Toeplitz 矩阵
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0, :].numpy()
    k_size = len(kernel)

    # 修正 Toeplitz 构造方式，确保维度匹配 seq_len
    first_col = np.zeros(seq_len)
    first_row = np.zeros(seq_len)
    first_col[:k_size // 2 + 1] = kernel[k_size // 2:]
    first_row[:k_size // 2 + 1] = kernel[k_size // 2::-1]
    toe_mat = toeplitz(first_col, first_row)

    if layer.last_W1 is not None:
        W1 = layer.last_W1[0].cpu().numpy()
        alpha, beta, theta = [torch.abs(getattr(weight_layer, p)).item() for p in ['alpha', 'beta', 'theta']]
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)

        # 3D 视图
        fig3d = plt.figure(figsize=(12, 10))
        ax3d = fig3d.add_subplot(111, projection='3d')
        W_smooth = gaussian_filter(zoom(W_combined, 2, order=3), sigma=0.5)
        X, Y = np.meshgrid(np.linspace(0, seq_len - 1, W_smooth.shape[0]),
                           np.linspace(0, seq_len - 1, W_smooth.shape[0]))
        surf = ax3d.plot_surface(X, Y, W_smooth, cmap=plt.cm.Blues, alpha=0.9, rcount=150, ccount=150)
        fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
        ax3d.view_init(elev=30, azim=250)
        plt.show()

        # 2D Magma 视图
        plt.figure(figsize=(10, 8))
        sns.heatmap(W_combined, cmap=plt.cm.Blues)
        plt.title("2D Fusion Matrix (Magma)")
        plt.show()


def visualize_network_topology(model):
    """
    绘制 ECG-KAN 模型的全局神经元连通图
    长相与 BCI 版本相同，自动适配 ECG 4层架构
    """
    model.eval()
    # 1. 自动提取所有 FastKANLayer 及其神经元配置
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, FastKANLayer):
            layers.append(m)

    if not layers:
        print("未在模型中找到 FastKANLayer 层。")
        return

    # 构造层级结构: [Input_Dim, L1_Out, L2_Out, L3_Out, L4_Out]
    layer_sizes = [layers[0].in_neurons] + [l.out_neurons for l in layers]
    num_layers = len(layer_sizes)

    # 2. 设置绘图坐标
    fig, ax = plt.subplots(figsize=(16, 10))
    pos = {}
    x_space = 2.5  # 增大层间距以便观察
    for l_idx, size in enumerate(layer_sizes):
        x = l_idx * x_space
        # 垂直居中排列神经元
        y_positions = np.linspace(-size / 2, size / 2, size)
        for n_idx, y in enumerate(y_positions):
            pos[(l_idx, n_idx)] = (x, y)

    # 3. 遍历每一层计算边能量并绘线
    all_lines = []
    colors = []

    with torch.no_grad():
        for l_idx, layer in enumerate(layers):
            # 能量 E = mean(|spline_weights|) + |omiga|
            w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
            tau = torch.abs(layer.tau)

            in_dim, out_dim = w_strength.shape
            for i in range(in_dim):
                for j in range(out_dim):
                    energy = w_strength[i, j].item()
                    threshold = tau[i, j].item()

                    # 核心逻辑：若能量 > 阈值，则视为导通
                    if energy > threshold:
                        start_pos = pos[(l_idx, i)]
                        end_pos = pos[(l_idx + 1, j)]
                        all_lines.append([start_pos, end_pos])
                        colors.append(energy)

    # 4. 使用 LineCollection 批量绘制边（Blues 配色）
    if all_lines:
        widths = 1.0 + 3.0 * (np.array(colors) / (max(colors) + 1e-8))
        lc = LineCollection(all_lines, cmap=plt.cm.Blues, array=np.array(colors),
                            linewidths=widths, alpha=0.8)
        line_obj = ax.add_collection(lc)
        # 添加颜色条
        cbar = fig.colorbar(line_obj, ax=ax)
        cbar.set_label(r'Edge Energy ($E = mean|W| + |\omega|$)')

    # 5. 绘制神经元节点 (白底黑边)
    for (l_idx, n_idx), (x, y) in pos.items():
        ax.scatter(x, y, s=200, c='white', edgecolors='black', zorder=3)

    # 6. 图表修饰
    ax.set_title("ECG-KAN Global Topology: Active Paths Audit", fontsize=16)
    ax.set_xlabel("Processing Stage (Layers)", fontsize=12)
    plt.xticks(np.arange(num_layers) * x_space,
               ['Input'] + [f'Layer {i + 1}' for i in range(num_layers - 1)])
    ax.get_yaxis().set_visible(False)
    plt.grid(False)
    plt.show()


def analyze_ecg_representation(model, dataloader, device, n_samples=2000):
    """
    非监督审计流程：分析输出层神经元是否真正分开了类别
    """
    model.eval()
    all_features = []
    all_labels = []
    collected = 0

    print(f">>> 正在提取 {n_samples} 个样本的输出层特征...")
    with torch.no_grad():
        for batch_data, batch_label in dataloader:
            # 提取 [B, 5, 187] 原始特征
            latent = model(batch_data.to(device), return_latent=True)

            # 1. 特征拉平 (Flatten): 每个样本变为 5 * 187 = 935 维向量
            flat_feat = latent.view(latent.size(0), -1)

            all_features.append(flat_feat.cpu().numpy())
            all_labels.append(batch_label.numpy())

            collected += batch_data.size(0)
            if collected >= n_samples: break

    # 汇总数据
    X = np.concatenate(all_features, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # 2. 数据标准化 (Standardization)
    X_scaled = StandardScaler().fit_transform(X)

    # 3. UMAP 流形降维 (2D)
    print(">>> 正在进行 UMAP 流形降维...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    # 4. HDBSCAN 非监督聚类
    print(">>> 正在进行 HDBSCAN 聚类分析...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)

    # 5. 可视化分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：基于真实标签上色 (查看模型分类能力)
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_true,
                           cmap='Spectral', s=10, alpha=0.6)
    ax1.set_title("Latent Space: Ground Truth Labels", fontsize=14)
    plt.colorbar(scatter1, ax=ax1, label='ECG Category (0-4)')

    # 右图：基于非监督聚类结果上色 (查看隐空间自然分布)
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels,
                           cmap='tab20', s=10, alpha=0.6)
    ax2.set_title("Latent Space: HDBSCAN Clusters", fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID (-1: Noise)')

    plt.suptitle("Output Layer Representation Analysis (N=5, L=187)", fontsize=16)
    plt.show()

    return embedding, cluster_labels


def print_layer_parameter_summary(model):
    """
    输出每一层的核心参数均值，并新增温度系数 T 的输出
    """
    # 更新表头，增加 Temperature (T) 列
    header = f"{'Layer Name':<12} | {'beta (W1)':>10} | {'theta (W3)':>10} | {'alpha (I)':>10} | {'gamma (Res)':>10} | {'omiga (Edge)':>12} | {'Temp (T)':>10}"
    print(header)
    print("-" * len(header))

    found_any = False
    for name, layer in model.named_modules():
        # 检查是否为包含审计参数的 FastKANLayer 层
        if hasattr(layer, 'weight_layers') and hasattr(layer, 'omiga') and hasattr(layer, 'temperature'):
            found_any = True

            # 1. 提取温度系数 T (在 forward 中通常取绝对值以保证数值意义)
            temp_val = torch.abs(layer.temperature).item()

            # 2. 计算 omiga 的均值
            omiga_mean = torch.mean(torch.abs(layer.omiga)).item()

            # 3. 遍历该层所有神经元提取 beta, theta, alpha, gamma
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

            # 打印结果，包含温度系数 T
            print(f"{name:<12} | {avg_beta:10.4f} | {avg_theta:10.4f} | {avg_alpha:10.4f} | {avg_gamma:10.4f} | {omiga_mean:12.4f} | {temp_val:10.4f}")

    if not found_any:
        print("!!! Warning: No FastKANLayer found. Check model structure.")


import torch


def count_parameters(model):
    """
    计算并打印模型的总参数量和可训练参数量
    """
    # 计算所有参数的总和
    total_params = sum(p.numel() for p in model.parameters())
    # 仅计算 requires_grad=True 的参数（用于迁移学习/冻结场景）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"==================================================")
    print(f"Model Summary:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"==================================================")

    return total_params, trainable_params


try:
    from thop import profile
    from thop import clever_format
except ImportError:
    print("提示: 未安装 thop 库，无法计算 FLOPs。请在终端运行 'pip install thop'")


def calculate_model_flops(model, device):
    """
    计算模型的 FLOPs (针对 1D ECG 信号)
    """
    model.eval()
    # 根据模型 forward 定义，构造输入 [1, 187] 或 [1, 1, 187]
    dummy_input = torch.randn(1, 187).to(device)

    try:
        # 使用 thop 进行分析
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

        # 格式化输出
        readable_flops, _ = clever_format([flops, 0], "%.3f")

        print(f"\n" + "=" * 50)
        print(f"模型计算复杂度审计 (Complexity Audit):")
        print(f"   总计 FLOPs:  {readable_flops}")
        print(f"   GFLOPs 单位: {flops / 1e9:.4f} G")
        print(f"=" * 50 + "\n")

        return flops
    except Exception as e:
        print(f"FLOPs 计算失败，原因: {e}")
        return 0


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    model = ECGKANModel(grid_size=64, spline_order=3).to(device)

    # 2. [新增] 仅输出 FLOPs
    calculate_model_flops(model, device)

    # 3. 后续原有逻辑 (加载权重等)...
    # count_parameters(model) # 如果不需要也可以注释掉


# # --- 测试运行 ---
# if __name__ == "__main__":
#     seed_everything(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 1. 初始化模型与测试数据
#     model = ECGKANModel(grid_size=64, spline_order=3).to(device)
#     test_csv_path = 'E:/database/archive/mitbih_test.csv'
#     val_loader = prepare_ecg_data(test_csv_path, batch_size=1)  # 设置为 1 方便审计单样本
#
#     # 2. 加载权重
#     weight_path = "D:\\treasure\\true\\thesis\\pth\\heart_wave\\v7\\heart_wave_1_9854.pth"
#     if os.path.exists(weight_path):
#         state_dict = torch.load(weight_path, map_location=device)
#         model.load_state_dict(state_dict, strict=False)
#
#     # 3. 提取样本并推理
#     # 3. 提取样本并推理 (修正版：直接索引 Dataset)
#     target_sample_idx = 4  # 修改这个数字即可切换样本
#
#     # 直接从 DataLoader 内部的 dataset 提取，不受 shuffle 影响
#     # tensors[0] 是数据, tensors[1] 是标签
#     sample_data = val_loader.dataset.tensors[0][target_sample_idx].unsqueeze(0)
#     sample_label = val_loader.dataset.tensors[1][target_sample_idx]
#
#     print(f"\n>>> 正在审计测试集第 {target_sample_idx} 个物理样本")
#     print(f">>> 样本标签: {sample_label.item()}")
#
#     target_layer_idx, target_neuron_idx = 3, 0
#     edge = (3, 6)
#     target_layer = getattr(model, f'layer{target_layer_idx}')
#     target_layer.visualize_idx = target_neuron_idx
#
#     count_parameters(model)
#     # print_layer_parameter_summary(model)
#
#     model.eval()
#     with torch.no_grad():
#         # 获取预处理后的特征用于对比图
#         prep_feat = model.prep_conv(sample_data.unsqueeze(1).to(device))
#         _ = model(sample_data.to(device))  # 激活 W1
#
#     audit_loader = prepare_ecg_data(test_csv_path, batch_size=64)
#
#     # 2. 执行分析流程
#     print("\n--- 启动输出层神经元表征审计 (UMAP + HDBSCAN) ---")
#     #analyze_ecg_representation(model, audit_loader, device, n_samples=3000)
#
#     # 4. 依次触发 8 个审计窗口
#     print("\n>>> 触发 8 窗口全量审计流程...")
#     # (1) W1 对齐分析
#     col_ratio = visualize_w1_analysis(target_layer, target_layer_idx, target_neuron_idx)
#     # (2) 原始/处理波形对比
#     visualize_waveforms(sample_data[0].numpy(), prep_feat[0].cpu().numpy())
#     # (5) 注意力波形
#     visualize_attention_waveforms_v2(sample_data[0].cpu().numpy(), prep_feat[0, 0].cpu().numpy(), col_ratio)
#     # (3) 激活函数映射
#     visualize_spline_mapping_window(target_layer, target_layer_idx, target_neuron_idx, edge)
#     # (4) 参数分布
#     # visualize_params_and_active_edges(target_layer, target_layer_idx, target_neuron_idx)
#     # (6 & 7) 3D 与 2D 融合矩阵
#     visualize_fusion_matrices(target_layer, target_layer_idx, target_neuron_idx)
#     # (8) 全局拓扑
#     # visualize_network_topology(model)
