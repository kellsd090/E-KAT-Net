import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from datapreprocessing_deap_ECG import DEAP_ECG_RegressionDataset, DEAP_GSR_RegressionDataset


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
    def __init__(self, in_hidden, out_hidden, kernel_size, in_features=200):
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
    def __init__(self, in_neurons, out_neurons, neuron_out_dim, grid_size=96, spline_order=3, hidden_dim=64,
                 kernel_size=15, seq_len=200):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order

        # 纯粹的 B-样条权重，去掉了 base_scales
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)

        if isinstance(neuron_out_dim, int):
            self.out_dims = [neuron_out_dim] * out_neurons
        else:
            self.out_dims = neuron_out_dim

        # 每一个输出神经元对应一个 1D 优化的 SplineWeightLayer
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_features=seq_len, in_hidden=hidden_dim, out_hidden=self.out_dims[j], kernel_size=kernel_size)
            for j in range(out_neurons)
        ])

        self.in_norm = nn.InstanceNorm1d(hidden_dim)

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
        x_reshaped = x_in.reshape(b * in_n, feat_dim, seq_len)
        x_normed = self.in_norm(x_reshaped)

        # 还原回 KAN 形状，缩放至 0.5 保护样条网格核心区
        x_norm = x_normed.reshape(b, in_n, seq_len, feat_dim)
        x_norm = torch.clamp(x_norm * 0.5, -0.99, 0.99)

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

        # --- 改良点 2：条件化 Stack ---
        # 如果所有神经元输出维度一致（Layer 1-4），则 stack 成 Tensor
        # 如果不一致（Layer 5），则返回列表，由下游 ECGKANModel 处理
        if all(o.shape == next_outputs[0].shape for o in next_outputs):
            return torch.stack(next_outputs, dim=1), torch.stack(next_projections, dim=1)
        else:
            return next_outputs, next_projections


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
    def __init__(self, kernel_size): # 7,17,31; 15,31,63;
        super(MultiScalePrepConv, self).__init__()
        # 三个并行分支，捕捉不同频率特征
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=kernel_size[0], padding=kernel_size[0]//2), # 抓高频细节
            nn.BatchNorm1d(8),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=kernel_size[1], padding=kernel_size[1]//2), # 抓标准形态
            nn.BatchNorm1d(20),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=kernel_size[2], padding=kernel_size[2]//2), # 抓长程节律(R-R间期)
            nn.BatchNorm1d(4),
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
    def __init__(self, kernel_size, ks=[7, 17, 31], grid_size=64, spline_order=3, seq_len=200): # 31,25,17,9,3; 63,47,31,15,7;
        super(ECGKANModel, self).__init__()
        self.prep_conv = MultiScalePrepConv(kernel_size=ks)
        # Layer 1 & 2: 保持 32 维特征
        self.layer1 = FastKANLayer(in_neurons=1, out_neurons=4, neuron_out_dim=64,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=32, kernel_size=kernel_size[0], seq_len=seq_len)

        self.layer2 = FastKANLayer(in_neurons=4, out_neurons=8, neuron_out_dim=96,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=64, kernel_size=kernel_size[1], seq_len=seq_len)

        self.layer3 = FastKANLayer(in_neurons=8, out_neurons=10, neuron_out_dim=64,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=96, kernel_size=kernel_size[2], seq_len=seq_len)

        self.layer4 = FastKANLayer(in_neurons=10, out_neurons=8, neuron_out_dim=32,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=64, kernel_size=kernel_size[3], seq_len=seq_len)

        # Layer 5 异构配置
        self.layer5 = FastKANLayer(in_neurons=8, out_neurons=5, neuron_out_dim=[1, 1, 1, 1, 32],
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=32, kernel_size=kernel_size[4], seq_len=seq_len)

    def forward(self, x, mode='independent', target_seq_len=None):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.prep_conv(x)  # [B, 32, 187]
        # 转换为 KAN 格式: [Batch, 1神经元, 187时间步, 32特征]
        x = x.transpose(1, 2).unsqueeze(1)

        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)  # x: [B, 5, 187, 1]
        x_list, proj5 = self.layer5(x, proj_x_prev=proj4)

        # 输出层：对 1 维特征求均值(保持不变)，再对时间 187 求平均
        # return torch.mean(x, dim=(-1, -2))
        if mode == 'independent':
            # --- 训练阶段 1：情感回归 ---
            # 1. x 现在是一个列表 [tens0, tens1, tens2, tens3, tens32]
            # 2. 提取前 4 个形状为 [B, 200, 1] 的神经元并堆叠为 [B, 4, 200, 1]
            x_emotion = torch.stack(x_list[:4], dim=1)
            # 3. 对时间步 (200) 和特征维 (1) 求平均，返回 [B, 4]
            return torch.mean(x_emotion, dim=(-1, -2))

        elif mode == 'cascade':
            # --- 训练阶段 2：桥接下一个神经簇 ---
            # 提取第 5 个神经元 [B, 200, 32]
            # 为了适配 Fusion 层输入格式，转换为 [B, 1, 200, 32]
            bridge_out = x_list[4].unsqueeze(1)
            if target_seq_len is not None:
                bridge_out = align_sequence_length(bridge_out, target_seq_len)
            return bridge_out

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

class AatyFusionModel(nn.Module):
    def __init__(self, expert_ecg, expert_gsr, fusion_hidden_dim=32, grid_size=32, seq_len=512):
        """
        expert_models: 一个 nn.ModuleList，包含已加载 pth 的专家模型 (如 ECG_Model, GSR_Model)
        """
        super(AatyFusionModel, self).__init__()
        self.expert_ecg = expert_ecg  # 预训练好的 ECG 专家
        self.expert_gsr = expert_gsr  # 预训练好的 GSR 专家
        self.experts = nn.ModuleList([expert_ecg, expert_gsr])

        # --- 新的融合神经簇 (几乎保持原有的 5 层结构) ---
        # 输入神经元数量 = 专家数量 (因为我们要融合每个专家的第 5 个神经元)
        self.fusion_layer1 = FastKANLayer(in_neurons=2, out_neurons=4,
                                          neuron_out_dim=64, grid_size=grid_size,
                                          hidden_dim=fusion_hidden_dim, seq_len=seq_len)

        self.fusion_layer2 = FastKANLayer(in_neurons=4, out_neurons=8,
                                          neuron_out_dim=96, grid_size=grid_size,
                                          hidden_dim=64, seq_len=seq_len)

        self.fusion_layer3 = FastKANLayer(in_neurons=8, out_neurons=10,
                                          neuron_out_dim=64, grid_size=grid_size,
                                          hidden_dim=96, seq_len=seq_len)

        self.fusion_layer4 = FastKANLayer(in_neurons=10, out_neurons=8,
                                          neuron_out_dim=32, grid_size=grid_size,
                                          hidden_dim=64, seq_len=seq_len)

        # 融合簇的最终输出 (4 个回归维度：V, A, D, L)
        self.fusion_layer5 = FastKANLayer(in_neurons=8, out_neurons=4,
                                          neuron_out_dim=1, grid_size=grid_size,
                                          hidden_dim=32, seq_len=seq_len)

    def forward(self, ecg_raw, gsr_raw):
        """
        inputs_list: 列表，例如 [ecg_tensor, gsr_tensor]
        """
        feat_ecg = self.expert_ecg(ecg_raw, mode='cascade', target_seq_len=512)
        feat_gsr = self.expert_gsr(gsr_raw, mode='cascade', target_seq_len=None)

        # 拼接专家特征作为融合簇的输入: [B, num_experts, seq_len, 32]
        x_fusion = torch.cat([feat_ecg, feat_gsr], dim=1)

        # 融合簇的前向传播
        x, proj1 = self.fusion_layer1(x_fusion, proj_x_prev=x_fusion)
        x, proj2 = self.fusion_layer2(x, proj_x_prev=proj1)
        x, proj3 = self.fusion_layer3(x, proj_x_prev=proj2)
        x, proj4 = self.fusion_layer4(x, proj_x_prev=proj3)
        # 融合簇的最后一层输出形状通常为 [B, 4, seq_len, 1]
        x_final, _ = self.fusion_layer5(x, proj_x_prev=proj4)

        # 对时间步和特征维求平均，得到 [B, 4]
        return torch.mean(x_final, dim=(-1, -2))

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


def align_sequence_length(tensor, target_len):
    """
    作用：将输入张量的时间维度重采样到目标长度。
    输入形状: [Batch, Neurons, Seq_Len, Hidden_Dim] -> 对应 KAN 的特征图
    输出形状: [Batch, Neurons, Target_Len, Hidden_Dim]
    """
    b, n, s, h = tensor.shape
    if s == target_len:
        return tensor

    # 1. 调整维度顺序以适配 interpolate [Batch * Neurons, Hidden_Dim, Seq_Len]
    # 我们把 Neurons 并入 Batch，把 Hidden_Dim 当作 Channel
    x = tensor.view(b * n, s, h).transpose(1, 2)

    # 2. 执行线性插值重采样
    # linear 模式对生理信号的连续性保持最好
    x = F.interpolate(x, size=target_len, mode='linear', align_corners=True)

    # 3. 还原形状 [Batch, Neurons, Target_Len, Hidden_Dim]
    return x.transpose(1, 2).view(b, n, target_len, h)


def setup_fusion_freeze_strategy(fusion_model):
    """
    精细化控制专家簇的参数冻结
    """
    # 1. 默认冻结所有专家模型的所有参数
    for expert in fusion_model.experts:
        expert.requires_grad_(False)

        # 2. 解冻指定的“解耦区”：
        # (1) Layer 4 全部神经元
        expert.layer4.requires_grad_(True)

        # (2) Layer 5 中属于“第 5 个神经元”的权重层
        # layer5.weight_layers[4] 对应第 5 个神经元内部的 SplineWeightLayer
        expert.layer5.weight_layers[4].requires_grad_(True)

        # (3) Layer 4 与 Layer 5 第 5 个神经元之间的“边”
        # 这些边存储在 FastKANLayer 的 spline_weights, omiga, tau 中
        # 我们只解冻索引为 [:, 4, :] 的部分
        expert.layer5.spline_weights.requires_grad = True
        expert.layer5.omiga.requires_grad = True
        expert.layer5.tau.requires_grad = True

        # 为了实现“仅解冻第 5 个神经元的输入边”，我们需要在梯度更新时加钩子或手动处理
        # 但在 requires_grad 层面，由于 Tensor 是整体，通常直接设为 True。
        # 建议在优化器中只针对特定的层进行操作（见下文）。

    # 3. 融合簇（新的神经簇）全量参与训练
    fusion_model.fusion_layer1.requires_grad_(True)
    fusion_model.fusion_layer2.requires_grad_(True)
    fusion_model.fusion_layer3.requires_grad_(True)
    fusion_model.fusion_layer4.requires_grad_(True)
    fusion_model.fusion_layer5.requires_grad_(True)

    print(">>> 融合模型参数策略设置完成：专家底层已冻结，高层连接已解开。")


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


# --- 测试运行 ---
if __name__ == "__main__":
    # 1. 实例化 grid_size=16 的模型
    # 这里的参数必须和产生那个 .pth 文件时完全一致
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGKANModel(grid_size=64, spline_order=3)

    # 请确保文件名是你之前保存 16 网格模型的文件名
    # 设置测试集文件路径
    test_csv_path = 'E:/database/archive/mitbih_test.csv'
    # 获取 val_loader
    # batch_size 建议与训练时保持一致（如 64）
    val_loader = DEAP_ECG_RegressionDataset(test_csv_path)

    state_dict = torch.load("D:/treasure/true/thesis/pth/heart_wave/v7/heart_wave_1_9851.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print("成功加载 grid_size=16 的模型权重")

    # 3. 调用你之前的可视化函数
    # ... 加载模型代码 ...
    if "model" in locals():
        # 1. 准备数据
        sample_data, sample_label = next(iter(val_loader))
        sample_data = sample_data.to(device)

        # --- 核心修正：设置可视化锚点 ---
        target_layer_idx = 3  # 你想看的层
        target_neuron_idx = 2  # 你想看的神经元

        # 必须先获取该层实例，并告诉它：等会儿 forward 时请记录第 4 个神经元的 W1
        target_layer = getattr(model, f'layer{target_layer_idx}')
        target_layer.visualize_idx = target_neuron_idx

        # 2. 执行一次前向传播，触发 W1 的计算与存储
        model.eval()
        with torch.no_grad():
            _ = model(sample_data)

            # 3. 此时 target_layer.last_W1 已经有值了
        edge = (1, 1)
        visualize_model_internals(model,
                                  layer_idx=target_layer_idx,
                                  neuron_idx=target_neuron_idx,
                                  edge_idx=edge)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可学习参数总量 (控制点): {total_params}")
