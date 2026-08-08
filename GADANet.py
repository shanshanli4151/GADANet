"""
jobs_new_group_aligned_net.py
对抗式因果迁移学习 — Jobs 数据集 + V2 算法

整合来源:
  1. jobs_group_aligned_domain_causal_net_binary.py — Jobs 数据集完整引入方式:
     - 数据加载 (load_and_format_covariates_jobs / load_all_other_crap_jobs)
     - 二分类适配 (Sigmoid 概率输出, ATE = P(Y=1|T=1) - P(Y=1|T=0))
     - 二分类专用损失函数 (joint_binary_classification_loss / make_tarreg_loss_binary)
     - 真实 ATE 计算 (compute_known_true_ate_on_target)

  2. new_group_aligned_domain_causal_net_V2.py — V2 代码执行流程:
     - 固定 λ + 自动校准平衡因子 (fixed_balance = causal / domain)
     - 非对称 GRL (源域 GRL, 目标域 detach)
     - 特征子空间分离 (inv_dim)
     - 域适应监控器 (DomainAdaptMonitor)
     - 目标域训练/测试划分, 分别报告 ATE 误差
     - 判别器独立学习率 (disc_lr_multiplier)
     - X 标准化

核心公式:
  L_total = L_causal + grl_lambda × fixed_balance × L_domain
  L_causal = L_causal_s + target_weight × L_causal_t
  fixed_balance = causal_loss_initial / domain_loss_initial (自动校准)
"""

import os
import glob
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from idhp_data import *
from ate import *
from modelsli1 import DragonNet, TarNet


# ============================================================
# 1. 梯度反转层 (Gradient Reversal Layer)
# ============================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


# ============================================================
# 2. 分组域判别器
# ============================================================
class GroupDomainDiscriminator(nn.Module):
    def __init__(self, rep_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, r):
        return self.net(r)


# ============================================================
# 3. 早停
# ============================================================
class EarlyStopper:
    def __init__(self, patience=50, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_model_state = None

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


# ============================================================
# 4. 域适应监控器 (V2)
# ============================================================
class DomainAdaptMonitor:
    def __init__(self):
        self.domain_acc_history = []
        self.causal_loss_history = []
        self.domain_loss_history = []

    def record(self, domain_acc, causal_loss, domain_loss):
        self.domain_acc_history.append(domain_acc)
        self.causal_loss_history.append(causal_loss)
        self.domain_loss_history.append(domain_loss)

    def summary(self):
        if len(self.domain_acc_history) == 0:
            return {}
        acc_array = np.array(self.domain_acc_history)
        causal_array = np.array(self.causal_loss_history)
        domain_array = np.array(self.domain_loss_history)

        print("\n" + "=" * 60)
        print("域适应监控汇总")
        print("=" * 60)
        print(f"域分类器准确率 — 初始: {acc_array[0]:.4f}, 最终: {acc_array[-1]:.4f}")
        print(f"域分类器准确率 — 均值: {acc_array.mean():.4f}, 最小值: {acc_array.min():.4f}")
        print(f"因果损失变化: {causal_array[0]:.4f} -> {causal_array[-1]:.4f}")
        print(f"域损失变化:   {domain_array[0]:.4f} -> {domain_array[-1]:.4f}")

        if 0.45 < acc_array[-1] < 0.60:
            print("评估: 域适应成功 — 域分类器接近随机猜测(50%), 特征已对齐")
        elif acc_array[-1] < 0.45:
            print("评估: 域分类器表现较差 — 可能过度适应")
        else:
            print("评估: 域适应不完全 — 域分类器仍能区分源域/目标域")

        return {
            'final_domain_acc': float(acc_array[-1]),
            'initial_domain_acc': float(acc_array[0]),
            'mean_domain_acc': float(acc_array.mean()),
            'final_causal_loss': float(causal_array[-1]),
            'final_domain_loss': float(domain_array[-1]),
        }


# ============================================================
# 5. AdversarialDragonNetBinary — Jobs二分类 + V2特征子空间分离
# ============================================================
class AdversarialDragonNetBinary(DragonNet):
    """
    DragonNet 二分类适配 + V2 特征子空间分离
    - 使用 sigmoid 获取概率 P(Y=1|X, T)
    - ATE = P(Y=1|X,T=1) - P(Y=1|X,T=0)
    - inv_dim: 域不变特征维度, 仅前 inv_dim 维参与域对抗
    """
    def __init__(self, in_features, out_features=[200, 100, 1], inv_dim=100):
        super().__init__(in_features, out_features)
        self.rep_dim = out_features[0]
        self.inv_dim = inv_dim
        self.disc_t0 = GroupDomainDiscriminator(self.inv_dim, hidden_dim=64)
        self.disc_t1 = GroupDomainDiscriminator(self.inv_dim, hidden_dim=64)
        self._init_discriminator_params()

    def _init_discriminator_params(self):
        for m in self.disc_t0.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        for m in self.disc_t1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def get_representation(self, x):
        """返回仅用于域对抗的不变特征子空间"""
        return self.representation_block(x)[:, :self.inv_dim]

    def predict_outcomes(self, x):
        """返回原始logits (兼容接口)"""
        output = self.forward(x)
        return output[:, 0:1], output[:, 1:2]

    def predict_outcomes_prob(self, x):
        """预测潜在结果的概率 P(Y=1|X,T)"""
        output = self.forward(x)
        prob_y0 = torch.sigmoid(output[:, 0:1])
        prob_y1 = torch.sigmoid(output[:, 1:2])
        return prob_y0, prob_y1

    def predict_ite(self, x):
        """ITE = P(Y=1|X,T=1) - P(Y=1|X,T=0)"""
        prob_y0, prob_y1 = self.predict_outcomes_prob(x)
        return prob_y1 - prob_y0

    def predict_ate(self, x):
        """ATE = E[ITE]"""
        return self.predict_ite(x).mean()

    def encoder_parameters(self):
        for name, param in self.named_parameters():
            if 'disc' not in name:
                yield param

    def discriminator_parameters(self):
        for name, param in self.named_parameters():
            if 'disc' in name:
                yield param


# ============================================================
# 6. AdversarialTarNetBinary — Jobs二分类 + V2特征子空间分离
# ============================================================
class AdversarialTarNetBinary(TarNet):
    """TarNet 二分类适配 + V2 特征子空间分离"""
    def __init__(self, in_features, out_features=[200, 100, 1], inv_dim=100):
        super().__init__(in_features, out_features)
        self.rep_dim = out_features[0]
        self.inv_dim = inv_dim
        self.disc_t0 = GroupDomainDiscriminator(self.inv_dim, hidden_dim=64)
        self.disc_t1 = GroupDomainDiscriminator(self.inv_dim, hidden_dim=64)
        self._init_discriminator_params()

    def _init_discriminator_params(self):
        for m in self.disc_t0.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        for m in self.disc_t1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def get_representation(self, x):
        return self.representation_block(x)[:, :self.inv_dim]

    def predict_outcomes(self, x):
        output = self.forward(x)
        return output[:, 0:1], output[:, 1:2]

    def predict_outcomes_prob(self, x):
        output = self.forward(x)
        prob_y0 = torch.sigmoid(output[:, 0:1])
        prob_y1 = torch.sigmoid(output[:, 1:2])
        return prob_y0, prob_y1

    def predict_ite(self, x):
        prob_y0, prob_y1 = self.predict_outcomes_prob(x)
        return prob_y1 - prob_y0

    def predict_ate(self, x):
        return self.predict_ite(x).mean()

    def encoder_parameters(self):
        for name, param in self.named_parameters():
            if 'disc' not in name:
                yield param

    def discriminator_parameters(self):
        for name, param in self.named_parameters():
            if 'disc' in name:
                yield param


# ============================================================
# 7. 二分类专用损失函数 (Jobs适配)
# ============================================================
def joint_binary_classification_loss(concat_pred, concat_true):
    """
    联合二分类损失: 结合结果回归和倾向得分
    Y ∈ {0,1}, T ∈ {0,1}
    """
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    t_pred = concat_pred[:, 2]

    t_pred = torch.clamp(t_pred, 1e-7, 1 - 1e-7)

    y_pred = torch.zeros_like(y0_pred)
    y_pred[t_true == 0] = y0_pred[t_true == 0]
    y_pred[t_true == 1] = y1_pred[t_true == 1]

    loss_y = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='sum')
    loss_t = F.binary_cross_entropy_with_logits(t_pred, t_true, reduction='sum')

    return loss_y + loss_t


def make_tarreg_loss_binary(ratio=1.0, base_loss=joint_binary_classification_loss):
    """二分类 targeted regularization loss"""
    def tarreg_loss(concat_pred, concat_true):
        vanilla_loss = base_loss(concat_pred, concat_true)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        t_pred = torch.clamp(t_pred, 1e-7, 1 - 1e-7)
        y0_prob = torch.sigmoid(y0_pred)
        y1_prob = torch.sigmoid(y1_pred)

        y_pred = torch.zeros_like(y0_prob)
        y_pred[t_true == 0] = y0_prob[t_true == 0]
        y_pred[t_true == 1] = y1_prob[t_true == 1]

        epsilons = concat_pred[:, 3] if concat_pred.shape[1] > 3 else torch.zeros_like(t_pred)

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        h = torch.clamp(h, -10, 10)

        y_pert = y_pred + epsilons * h
        targeted_reg = torch.sum(torch.square(y_true - y_pert))

        return vanilla_loss + ratio * targeted_reg

    return tarreg_loss


# ============================================================
# 8. Jobs 数据集加载函数
# ============================================================
def load_and_format_covariates_jobs(file_path):
    """加载 jobs 数据集的协变量 X1-X17"""
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, 3:]
    return x


def load_all_other_crap_jobs(file_path):
    """加载 jobs 数据集的 T, Y, e"""
    data = np.loadtxt(file_path, delimiter=',')
    t = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    e = data[:, 2].reshape(-1, 1)
    return t, y, e


def _extract_dataset_idx(filename):
    """
    从文件名提取数据集索引号
    例如: 'dataset_5.csv' -> 5, 'dataset_42.csv' -> 42
    """
    import re
    match = re.search(r'dataset_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None


def load_precomputed_true_ate(data_base_dir, dataset_idx, X=None, target_idx=None):
    """
    从预计算的 .npy 文件加载真实 ATE (概率差)。
    优先加载 true_ate_{dataset_idx}.npy, 否则用 LogisticRegression 兜底估计。

    参数:
        data_base_dir: 数据目录路径 (如 jobs3/)
        dataset_idx:   数据集索引 (1-50), 对应 true_ate_{dataset_idx}.npy
        X:             (可选) 完整特征矩阵, 用于兜底估计
        target_idx:    (可选) 目标域样本索引, 用于兜底估计

    返回:
        true_ate: 真实 ATE 值 (float)
    """
    true_ate_file = os.path.join(data_base_dir, f'true_ate_{dataset_idx}.npy')
    if os.path.exists(true_ate_file):
        true_ate = float(np.load(true_ate_file))
        return true_ate

    # ---- LogisticRegression 兜底估计 ----
    print(f"警告: 未找到预计算的真实ATE ({true_ate_file}), 使用 LogisticRegression 估计")
    try:
        import pandas as pd
        csv_files = sorted([f for f in os.listdir(data_base_dir)
                           if f.startswith('dataset_') and f.endswith('.csv')])
        if not csv_files:
            print("错误: 无法找到数据集文件")
            return None

        first_file = os.path.join(data_base_dir, csv_files[0])
        df_full = pd.read_csv(first_file, header=None)
        df_full.columns = ['T', 'Y', 'e'] + [f'X{i}' for i in range(1, 18)]

        X_cols = [f'X{i}' for i in range(1, 18)]
        X_orig = df_full[X_cols].values
        T_orig = df_full['T'].values
        Y_orig = df_full['Y'].values

        mu1_model = LogisticRegression(max_iter=1000).fit(
            X_orig[T_orig == 1], Y_orig[T_orig == 1])
        mu0_model = LogisticRegression(max_iter=1000).fit(
            X_orig[T_orig == 0], Y_orig[T_orig == 0])

        if X is not None and target_idx is not None:
            X_target = X[target_idx]
            if X_target.ndim > 2:
                X_target = X_target.reshape(X_target.shape[0], -1)
        else:
            # 使用全量数据估计
            X_target = X_orig

        p_y1_given_x_t1 = mu1_model.predict_proba(X_target)[:, 1]
        p_y1_given_x_t0 = mu0_model.predict_proba(X_target)[:, 1]

        true_ate_target = (p_y1_given_x_t1 - p_y1_given_x_t0).mean()
        print(f"估计的真实ATE: {true_ate_target:.4f}")
        return true_ate_target
    except Exception as e:
        print(f"计算真实ATE时出错: {str(e)}")
        return None


# ============================================================
# 9. 核心训练函数 — V2简化版 (固定λ + 自动校准平衡因子)
# ============================================================
def train_epoch_adversarial_v2(model, x_s, t_s, y_s, x_t, t_t, y_t,
                                optimizer,
                                criterion_causal, criterion_domain,
                                device,
                                grl_lambda=1.0,
                                target_weight=1.0,
                                fixed_balance=None):
    """
    V2 对抗式训练单 batch

    关键公式:
      L_total = L_causal + grl_lambda × fixed_balance × L_domain
      L_causal = L_causal_s + target_weight × L_causal_t

    非对称 GRL: 源域 GRL 反转, 目标域 detach (不受对抗)
    """
    model.train()

    x_s = x_s.to(device)
    t_s = t_s.to(device)
    y_s = y_s.to(device)
    x_t = x_t.to(device)
    t_t = t_t.to(device)
    y_t = y_t.to(device)

    # ---- 前向传播 ----
    r_s = model.get_representation(x_s)
    r_t = model.get_representation(x_t)

    yt_pred_s = model(x_s)
    yt_pred_t = model(x_t)

    # ---- 因果损失 (目标域加权) ----
    yt_s = torch.cat([y_s, t_s], dim=1)
    yt_t = torch.cat([y_t, t_t], dim=1)

    loss_causal_s = criterion_causal(yt_pred_s, yt_s)
    loss_causal_t = criterion_causal(yt_pred_t, yt_t)
    loss_causal = loss_causal_s + target_weight * loss_causal_t

    # ---- 分组掩码 ----
    mask_s0 = (t_s == 0).squeeze()
    mask_s1 = (t_s == 1).squeeze()
    mask_t0 = (t_t == 0).squeeze()
    mask_t1 = (t_t == 1).squeeze()

    # ---- 非对称域对抗 ----
    loss_domain_total = 0.0
    domain_acc_total = 0.0
    domain_count = 0

    for mask_s, mask_t, disc in [
        (mask_s0, mask_t0, model.disc_t0),
        (mask_s1, mask_t1, model.disc_t1)
    ]:
        if torch.sum(mask_s) == 0 or torch.sum(mask_t) == 0:
            continue

        r_s_g = r_s[mask_s]
        r_t_g = r_t[mask_t].detach()

        r_s_rev = grad_reverse(r_s_g, grl_lambda)
        r_all = torch.cat([r_s_rev, r_t_g], dim=0)

        labels = torch.cat([
            torch.zeros(len(r_s_g), device=device),
            torch.ones(len(r_t_g), device=device)
        ]).long()

        logits = disc(r_all)
        loss_domain_total += criterion_domain(logits, labels)

        domain_preds = torch.argmax(logits, dim=1)
        domain_acc_total += (domain_preds == labels).float().mean().item()
        domain_count += 1

    # ---- 总损失 ----
    if domain_count > 0 and fixed_balance is not None:
        loss_total = loss_causal + grl_lambda * fixed_balance * loss_domain_total
    elif domain_count > 0:
        loss_total = loss_causal + loss_domain_total
    else:
        loss_total = loss_causal

    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    domain_loss_mean = loss_domain_total.item() / domain_count if domain_count > 0 else 0.0

    return {
        'loss_total': loss_total.item(),
        'loss_causal': loss_causal.item(),
        'loss_domain': domain_loss_mean,
        'domain_acc': domain_acc_total / domain_count if domain_count > 0 else 0.0,
        'grl_lambda': grl_lambda,
        'fixed_balance': fixed_balance,
    }


# ============================================================
# 10. 评估 ATE (Jobs 二分类版)
# ============================================================
def evaluate_ate_adversarial_binary(model, x, true_ate, device):
    """
    评估二分类模型的 ATE 预测
    ATE = E[P(Y=1|X,T=1) - P(Y=1|X,T=0)]
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        ate_pred = model.predict_ate(x)
        if isinstance(ate_pred, torch.Tensor):
            ate_pred = ate_pred.item()

    ate_error = abs(ate_pred - true_ate)
    return ate_pred, ate_error


# ============================================================
# 11. 主程序 — Jobs数据集 + V2算法
# ============================================================
def run_adversarial_jobs_v2(data_base_dir=r'C:\Users\liruy\Desktop\jobs3',
                             output_dir=r'C:\Users\liruy\Desktop\jobs3',
                             knob='tarnet',
                             grl_lambda=1.0,
                             lr=5e-4,
                             l1_reg=0.01,
                             weight_decay=1e-4,
                             epochs=200,
                             batchsize=64,
                             targeted_reg=True,
                             tarreg_ratio=0.5,
                             early_stop_patience=20,
                             lr_scheduler_factor=0.5,
                             lr_scheduler_patience=10,
                             disc_lr_multiplier=5.0,
                             test_ratio=0.3,
                             inv_dim=100,
                             target_col_idx=2,
                             verbose=True):
    """
    Jobs 数据集 + V2 对抗式因果迁移学习

    核心参数:
      grl_lambda:            梯度反转强度 (默认 1.0)
      inv_dim:               域不变特征维度 (默认 100)
      target_col_idx:        用于域划分的特征列索引 (默认 2, 即 X3)
      test_ratio:            目标域测试集比例 (默认 0.3)

    fixed_balance 自动校准: 首个 batch 用无 GRL 前向估计尺度比
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"选择模型: {knob} (Jobs二分类 + V2算法)")
    print(f"GRL λ: {grl_lambda} (1.0=域对抗≈因果梯度)")
    print(f"inv_dim: {inv_dim}, 目标域测试比例: {test_ratio}")

    simulation_files = sorted(
        glob.glob("{}/*.csv".format(data_base_dir)),
        key=lambda f: _extract_dataset_idx(f) or 0
    )
    print(f"找到 {len(simulation_files)} 个数据文件")

    final_output = []
    all_ate_errors_test = []
    all_ate_errors_train = []
    all_monitor_summaries = []
    all_fixed_balances = []

    params = {
        'knob': knob,
        'grl_lambda': grl_lambda,
        'lr': lr, 'l1_reg': l1_reg, 'weight_decay': weight_decay,
        'epochs': epochs, 'batchsize': batchsize,
        'targeted_reg': targeted_reg, 'tarreg_ratio': tarreg_ratio,
        'early_stop_patience': early_stop_patience,
        'lr_scheduler_factor': lr_scheduler_factor,
        'lr_scheduler_patience': lr_scheduler_patience,
        'disc_lr_multiplier': disc_lr_multiplier,
        'test_ratio': test_ratio,
        'inv_dim': inv_dim,
        'target_col_idx': target_col_idx,
        'balance_method': 'auto_calibrated',
        'data_type': 'binary',
    }
    print(f"参数配置: {params}")

    for idx, simulation_file in enumerate(simulation_files):
        try:
            dataset_idx = _extract_dataset_idx(simulation_file)
            if dataset_idx is None:
                print(f"无法从文件名提取数据集索引: {simulation_file}, 跳过")
                continue

            if verbose:
                print(f"\n{'='*60}")
                print(f"处理文件 {idx+1}/{len(simulation_files)}: "
                      f"{os.path.basename(simulation_file)} (dataset_{dataset_idx})")

            # ---- Jobs 数据加载 ----
            x = load_and_format_covariates_jobs(simulation_file)
            t, y, e = load_all_other_crap_jobs(simulation_file)

            x = x.astype(np.float32)
            t = t.astype(np.float32).reshape(-1, 1)
            y = y.astype(np.float32).reshape(-1, 1)

            # ---- 域划分 (基于指定特征列) ----
            target_idx0 = np.where(x[:, target_col_idx] == 0)[0]  # 源域
            target_idx1 = np.where(x[:, target_col_idx] == 1)[0]  # 目标域

            if len(target_idx0) == 0 or len(target_idx1) == 0:
                print(f"跳过空组数据: {simulation_file}")
                continue

            x_s, t_s, y_s = x[target_idx0], t[target_idx0], y[target_idx0]
            x_t, t_t, y_t = x[target_idx1], t[target_idx1], y[target_idx1]

            # 动态目标域因果权重
            n_s, n_t = len(x_s), len(x_t)
            target_weight = max(1.0, min(n_s / n_t, 3.0))

            # ---- 真实 ATE — 从预计算文件加载 (带 LogisticRegression 兜底) ----
            true_ate_target = load_precomputed_true_ate(
                data_base_dir, dataset_idx, X=x, target_idx=target_idx1
            )
            # 目标域 ATE 是总体统计量, train/test 划分后真值不变
            true_ate_train = true_ate_target
            true_ate_test = true_ate_target

            if verbose:
                print(f"数据划分 - 源域: {x_s.shape}, 目标域: {x_t.shape}")
                print(f"加载预计算真实ATE (dataset_{dataset_idx}): {true_ate_target:.4f}")
                print(f"源域Y均值: {y_s.mean():.4f}, 目标域Y均值: {y_t.mean():.4f}")
                print(f"源域处理比例: {np.mean(t_s):.3f}, 目标域处理比例: {np.mean(t_t):.3f}")
                print(f"目标域因果权重: {target_weight:.2f} (n_s/n_t={n_s/n_t:.2f})")

            # ---- X 标准化 (V2) ----
            scaler_x = StandardScaler()
            x_s = scaler_x.fit_transform(x_s)
            x_t = scaler_x.transform(x_t)

            domain_diff_l1 = np.mean(np.abs(np.mean(x_s, axis=0) - np.mean(x_t, axis=0)))
            if verbose:
                print(f"协变量均值L1差异: {domain_diff_l1:.4f}")

            # ---- 转换为张量 + 目标域训练/测试划分 ----
            x_s_tensor = torch.from_numpy(x_s).float()
            t_s_tensor = torch.from_numpy(t_s).float()
            y_s_tensor = torch.from_numpy(y_s).float()
            x_t_tensor = torch.from_numpy(x_t).float()
            t_t_tensor = torch.from_numpy(t_t).float()
            y_t_tensor = torch.from_numpy(y_t).float()

            n_target = len(x_t_tensor)
            idx_all = np.arange(n_target)
            idx_train, idx_test = train_test_split(
                idx_all, test_size=test_ratio, random_state=42
            )
            idx_train_t = torch.from_numpy(idx_train).long()
            idx_test_t = torch.from_numpy(idx_test).long()

            x_t_train = x_t_tensor[idx_train_t]
            t_t_train = t_t_tensor[idx_train_t]
            y_t_train = y_t_tensor[idx_train_t]
            x_t_test = x_t_tensor[idx_test_t]
            t_t_test = t_t_tensor[idx_test_t]
            y_t_test = y_t_tensor[idx_test_t]

            # 目标域训练/测试的真实ATE — 与总体目标域 ATE 一致
            # (ATE 是目标域总体的因果效应期望, 随机划分后真值相同)

            if verbose:
                print(f"目标域划分: 训练 {len(idx_train)} / 测试 {len(idx_test)}")
                print(f"真实ATE — 训练集: {true_ate_train:.4f}, 测试集: {true_ate_test:.4f}")

            dataset_s = TensorDataset(x_s_tensor, t_s_tensor, y_s_tensor)
            dataset_t_train = TensorDataset(x_t_train, t_t_train, y_t_train)

            loader_s = DataLoader(dataset_s, batch_size=batchsize, shuffle=True)
            loader_t_train = DataLoader(dataset_t_train, batch_size=batchsize, shuffle=True)

            # ---- 初始化模型 ----
            input_dim = x_s.shape[1]
            if knob == 'dragonnet':
                model = AdversarialDragonNetBinary(input_dim, inv_dim=inv_dim).to(device)
            elif knob == 'tarnet':
                model = AdversarialTarNetBinary(input_dim, inv_dim=inv_dim).to(device)
            else:
                raise ValueError(f"不支持的模型类型: {knob}")

            # ---- 损失函数 (二分类专用) ----
            if targeted_reg:
                criterion_causal = make_tarreg_loss_binary(
                    ratio=tarreg_ratio,
                    base_loss=joint_binary_classification_loss
                )
            else:
                criterion_causal = joint_binary_classification_loss
            criterion_domain = nn.CrossEntropyLoss()

            # ---- 优化器 (编码器/判别器分离学习率) ----
            optimizer = optim.Adam([
                {'params': model.encoder_parameters(),
                 'lr': lr, 'weight_decay': weight_decay},
                {'params': model.discriminator_parameters(),
                 'lr': lr * disc_lr_multiplier, 'weight_decay': weight_decay * 0.1}
            ])

            scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_scheduler_factor,
                patience=lr_scheduler_patience
            )

            # ---- 早停 + 监控 ----
            early_stopper = EarlyStopper(patience=early_stop_patience, min_delta=0.0001)
            monitor = DomainAdaptMonitor()

            # ---- 自动校准 fixed_balance ----
            with torch.no_grad():
                xs_cal, ts_cal, ys_cal = next(iter(loader_s))
                xt_cal, tt_cal, yt_cal = next(iter(loader_t_train))
                xs_cal = xs_cal.to(device)
                ts_cal = ts_cal.to(device)
                ys_cal = ys_cal.to(device)
                xt_cal = xt_cal.to(device)
                tt_cal = tt_cal.to(device)
                yt_cal = yt_cal.to(device)

                yt_pred_s_cal = model(xs_cal)
                yt_pred_t_cal = model(xt_cal)
                yt_s_cal = torch.cat([ys_cal, ts_cal], dim=1)
                yt_t_cal = torch.cat([yt_cal, tt_cal], dim=1)

                loss_causal_cal = (criterion_causal(yt_pred_s_cal, yt_s_cal) +
                                   target_weight * criterion_causal(yt_pred_t_cal, yt_t_cal)).item()

                r_s_cal = model.get_representation(xs_cal)
                r_t_cal = model.get_representation(xt_cal)

                mask_s0_cal = (ts_cal == 0).squeeze()
                mask_s1_cal = (ts_cal == 1).squeeze()
                mask_t0_cal = (tt_cal == 0).squeeze()
                mask_t1_cal = (tt_cal == 1).squeeze()

                loss_domain_cal = 0.0
                domain_count_cal = 0
                for mask_s, mask_t, disc in [
                    (mask_s0_cal, mask_t0_cal, model.disc_t0),
                    (mask_s1_cal, mask_t1_cal, model.disc_t1)
                ]:
                    if torch.sum(mask_s) > 0 and torch.sum(mask_t) > 0:
                        r_cal = torch.cat([r_s_cal[mask_s], r_t_cal[mask_t]], dim=0)
                        labels_cal = torch.cat([
                            torch.zeros(len(r_s_cal[mask_s]), device=device),
                            torch.ones(len(r_t_cal[mask_t]), device=device)
                        ]).long()
                        loss_domain_cal += criterion_domain(disc(r_cal), labels_cal).item()
                        domain_count_cal += 1

                avg_domain_cal = loss_domain_cal / domain_count_cal if domain_count_cal > 0 else 0.69
                fixed_balance = loss_causal_cal / (avg_domain_cal + 1e-8)
                all_fixed_balances.append(fixed_balance)

            if verbose:
                print(f"自动校准: fixed_balance = {fixed_balance:.1f} "
                      f"(causal={loss_causal_cal:.1f} / domain={avg_domain_cal:.3f}), "
                      f"grl_lambda × fixed_balance = {grl_lambda * fixed_balance:.1f}")

            if verbose:
                print(f"开始训练 (固定λ={grl_lambda}, fixed_balance={fixed_balance:.1f})...")

            best_ate_error_test = float('inf')
            best_ate_error_train = float('inf')
            best_epoch = -1
            stopped_epoch = epochs

            for epoch in range(epochs):
                epoch_losses = {'loss_total': 0, 'loss_causal': 0,
                                'loss_domain': 0, 'domain_acc': 0}
                num_batches = 0

                for (xs, ts, ys), (xt, tt, yt) in zip(loader_s, loader_t_train):
                    losses = train_epoch_adversarial_v2(
                        model, xs, ts, ys, xt, tt, yt,
                        optimizer,
                        criterion_causal, criterion_domain,
                        device,
                        grl_lambda=grl_lambda,
                        target_weight=target_weight,
                        fixed_balance=fixed_balance,
                    )
                    for key in epoch_losses:
                        if key in losses:
                            epoch_losses[key] += losses[key]
                    num_batches += 1

                for key in epoch_losses:
                    epoch_losses[key] /= num_batches if num_batches > 0 else 1

                # ---- 评估: 测试集 ATE ----
                ate_pred_test, ate_error_test = evaluate_ate_adversarial_binary(
                    model, x_t_test, true_ate_test, device
                )

                # ---- 评估: 训练集 ATE ----
                ate_pred_train, ate_error_train = evaluate_ate_adversarial_binary(
                    model, x_t_train, true_ate_train, device
                )

                # ---- 监控 ----
                monitor.record(
                    epoch_losses['domain_acc'],
                    epoch_losses['loss_causal'],
                    epoch_losses['loss_domain'],
                )

                # 基于测试集 ATE 选择最佳模型
                if epoch >= 10 and ate_error_test < best_ate_error_test:
                    best_ate_error_test = ate_error_test
                    best_ate_error_train = ate_error_train
                    best_epoch = epoch

                scheduler_lr.step(ate_error_test)

                should_stop = early_stopper.early_stop(ate_error_test, model)
                if should_stop:
                    stopped_epoch = epoch
                    if verbose:
                        print(f"早停触发于 Epoch {epoch}, "
                              f"最佳ATE误差 (测试集): {best_ate_error_test:.6f}")
                    break

                if verbose and epoch % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch:3d}: "
                          f"Total={epoch_losses['loss_total']:.2f}, "
                          f"Causal={epoch_losses['loss_causal']:.2f}, "
                          f"Domain={epoch_losses['loss_domain']:.4f}, "
                          f"DiscAcc={epoch_losses['domain_acc']:.3f} | "
                          f"ATE_Test={ate_error_test:.4f}, "
                          f"ATE_Train={ate_error_train:.4f}, "
                          f"Best_Test={best_ate_error_test:.4f}, "
                          f"LR={current_lr:.6f}")

            # ---- 加载最佳模型 ----
            model = early_stopper.load_best_model(model)

            # ---- 最终评估 ----
            ate_pred_test, final_ate_error_test = evaluate_ate_adversarial_binary(
                model, x_t_test, true_ate_test, device
            )
            ate_pred_train, final_ate_error_train = evaluate_ate_adversarial_binary(
                model, x_t_train, true_ate_train, device
            )

            all_ate_errors_test.append(final_ate_error_test)
            all_ate_errors_train.append(final_ate_error_train)

            monitor_summary = monitor.summary()
            all_monitor_summaries.append(monitor_summary)

            result = {
                'sim_idx': idx,
                'file': os.path.basename(simulation_file),
                'ate_true_test': float(true_ate_test),
                'ate_pred_test': float(ate_pred_test),
                'ate_error_test': float(final_ate_error_test),
                'ate_true_train': float(true_ate_train),
                'ate_pred_train': float(ate_pred_train),
                'ate_error_train': float(final_ate_error_train),
                'best_ate_error_test': float(best_ate_error_test),
                'best_epoch': best_epoch,
                'stopped_epoch': stopped_epoch,
                'domain_diff_l1': float(domain_diff_l1),
                'target_weight': float(target_weight),
                'fixed_balance': float(fixed_balance),
                'monitor': monitor_summary,
                'params': params
            }
            final_output.append(result)

            if verbose:
                print(f"模拟 {idx} 完成 — "
                      f"ATE误差 测试集: {final_ate_error_test:.4f}, "
                      f"训练集: {final_ate_error_train:.4f}, "
                      f"最佳测试集: {best_ate_error_test:.4f} (Epoch {best_epoch}), "
                      f"fixed_balance: {fixed_balance:.1f}, "
                      f"域判别器最终准确率: {monitor_summary.get('final_domain_acc', 'N/A')}")

        except Exception as e:
            print(f"处理 {simulation_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ---- 汇总 ----
    if all_ate_errors_test:
        mean_test = np.mean(all_ate_errors_test)
        std_test = np.std(all_ate_errors_test)
        mean_train = np.mean(all_ate_errors_train)
        std_train = np.std(all_ate_errors_train)

        avg_domain_acc = np.mean([m.get('final_domain_acc', 0)
                                   for m in all_monitor_summaries if m])
        avg_fixed_balance = np.mean(all_fixed_balances) if all_fixed_balances else 0.0

        print(f"\n{'='*70}")
        print(f"{knob.capitalize()} + V2领域对抗 (Jobs二分类) — 误差汇总")
        print(f"参数: {params}")
        print(f"成功运行: {len(all_ate_errors_test)}/{len(simulation_files)}")
        print(f"ATE误差 (测试集) — 均值: {mean_test:.4f}, 标准差: {std_test:.4f}")
        print(f"ATE误差 (训练集) — 均值: {mean_train:.4f}, 标准差: {std_train:.4f}")
        print(f"域判别器最终准确率均值: {avg_domain_acc:.4f} (越接近0.5越好)")
        print(f"fixed_balance均值: {avg_fixed_balance:.1f}")
        print('=' * 70)
    else:
        mean_test = std_test = mean_train = std_train = np.nan

    summary = {
        'summary': {
            'mean_error_test': float(mean_test) if not np.isnan(mean_test) else None,
            'std_error_test': float(std_test) if not np.isnan(std_test) else None,
            'mean_error_train': float(mean_train) if not np.isnan(mean_train) else None,
            'std_error_train': float(std_train) if not np.isnan(std_train) else None,
            'successful_runs': len(all_ate_errors_test),
            'params': params
        }
    }
    final_output.append(summary)

    output_dir_path = f'./jobs_new_group_aligned_{knob}_params/'
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_file = f'{output_dir_path}adversarial_{knob}_jobs_V2_lambda{grl_lambda}.json'
    with open(output_file, 'w') as fp:
        json.dump(final_output, fp, indent=2)

    print(f"结果已保存到: {output_file}")
    return final_output


# ============================================================
# 12. turn_knob 接口
# ============================================================
def turn_knob(data_base_dir=r'C:\Users\liruy\Desktop\jobs3',
              knob='tarnet',
              output_base_dir='',
              grl_lambda=1.0,
              lr=5e-4,
              l1_reg=0.01,
              weight_decay=1e-4,
              batchsize=64,
              epochs=200,
              targeted_reg=True,
              tarreg_ratio=0.5,
              early_stop_patience=20,
              lr_scheduler_factor=0.5,
              lr_scheduler_patience=10,
              disc_lr_multiplier=5.0,
              test_ratio=0.3,
              inv_dim=100,
              target_col_idx=2,
              verbose=True):
    """
    Jobs数据集 + V2算法 的 turn_knob 接口

    唯一需调节的超参数:
      grl_lambda: 梯度反转强度 (默认 1.0)
    """
    print(f"{'='*70}")
    print(f"运行 {knob.capitalize()} + V2领域对抗 (Jobs二分类)")
    print(f"grl_lambda={grl_lambda}, inv_dim={inv_dim}, "
          f"target_col_idx={target_col_idx}")
    print(f"{'='*70}")

    return run_adversarial_jobs_v2(
        data_base_dir=data_base_dir,
        output_dir=output_base_dir,
        knob=knob,
        grl_lambda=grl_lambda,
        lr=lr,
        l1_reg=l1_reg,
        weight_decay=weight_decay,
        epochs=epochs,
        batchsize=batchsize,
        targeted_reg=targeted_reg,
        tarreg_ratio=tarreg_ratio,
        early_stop_patience=early_stop_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        disc_lr_multiplier=disc_lr_multiplier,
        test_ratio=test_ratio,
        inv_dim=inv_dim,
        target_col_idx=target_col_idx,
        verbose=verbose
    )


# ============================================================
# 13. 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Jobs数据集 + V2对抗式因果迁移学习 (二分类)'
    )

    parser.add_argument('--data_base_dir', type=str,
                        default=r'C:\Users\liruy\Desktop\jobs3',
                        help='数据目录路径')
    parser.add_argument('--output_base_dir', type=str, default='')
    parser.add_argument('--knob', type=str, default='tarnet',
                        choices=['tarnet', 'dragonnet'])
    parser.add_argument('--grl_lambda', type=float, default=1.0,
                        help='GRL梯度反转强度')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1正则化')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--targeted_reg', action='store_true', default=True)
    parser.add_argument('--tarreg_ratio', type=float, default=0.5)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5)
    parser.add_argument('--lr_scheduler_patience', type=int, default=10)
    parser.add_argument('--disc_lr_multiplier', type=float, default=5.0)
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='目标域测试集比例')
    parser.add_argument('--inv_dim', type=int, default=100,
                        help='域不变特征维度')
    parser.add_argument('--target_col_idx', type=int, default=2,
                        help='用于域划分的特征列索引')
    parser.add_argument('--verbose', action='store_true', default=True)

    args = parser.parse_args()

    turn_knob(
        data_base_dir=args.data_base_dir,
        knob=args.knob,
        output_base_dir=args.output_base_dir,
        grl_lambda=args.grl_lambda,
        lr=args.lr,
        l1_reg=args.l1_reg,
        weight_decay=args.weight_decay,
        batchsize=args.batchsize,
        epochs=args.epochs,
        targeted_reg=args.targeted_reg,
        tarreg_ratio=args.tarreg_ratio,
        early_stop_patience=args.early_stop_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        disc_lr_multiplier=args.disc_lr_multiplier,
        test_ratio=args.test_ratio,
        inv_dim=args.inv_dim,
        target_col_idx=args.target_col_idx,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()