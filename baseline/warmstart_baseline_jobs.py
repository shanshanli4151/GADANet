"""
Warmstart对比迁移模型 - 无对抗适应版本（适用于JOBS数据集）

该模型实现两阶段训练流程：
1. 第一阶段：使用源域数据训练TarNet/DragonNet模型，保存预训练参数
2. 第二阶段：使用目标域数据，基于预训练模型进行热启动训练

技术特点：
- 仅包含TarNet和DragonNet核心模型
- 无对抗域适应组件
- 支持模型参数保存与加载
- 针对JOBS数据集的二分类Y变量进行适配
"""

import os
import glob
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from modelsli1 import DragonNet, TarNet


# ------------------------------------------------------------
# 1. 早停类
# ------------------------------------------------------------
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0001):
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


# ------------------------------------------------------------
# 2. 二分类损失函数
# ------------------------------------------------------------
def joint_binary_classification_loss(concat_pred, concat_true):
    """联合二分类损失：结合结果回归和倾向得分"""
    y_true = concat_true[:, 0]  # [batch]
    t_true = concat_true[:, 1]  # [batch]
    
    y0_pred = concat_pred[:, 0]  # [batch]
    y1_pred = concat_pred[:, 1]  # [batch]
    t_pred = concat_pred[:, 2]   # [batch]
    
    t_pred = torch.clamp(t_pred, 1e-7, 1 - 1e-7)
    
    # 使用条件索引选择
    y_pred = torch.zeros_like(y0_pred)
    y_pred[t_true == 0] = y0_pred[t_true == 0]
    y_pred[t_true == 1] = y1_pred[t_true == 1]
    
    # Y是二分类0/1，使用binary_cross_entropy_with_logits
    loss_y = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='sum')
    
    # T是二分类0/1
    loss_t = F.binary_cross_entropy_with_logits(t_pred, t_true, reduction='sum')
    
    return loss_y + loss_t


def make_tarreg_loss_binary(ratio=1.0, base_loss=joint_binary_classification_loss):
    """针对二分类结果的targeted regularization loss"""
    def tarreg_loss(concat_pred, concat_true):
        vanilla_loss = base_loss(concat_pred, concat_true)
        
        y_true = concat_true[:, 0]  # [batch]
        t_true = concat_true[:, 1]  # [batch]
        
        y0_pred = concat_pred[:, 0]  # [batch]
        y1_pred = concat_pred[:, 1]  # [batch]
        t_pred = concat_pred[:, 2]   # [batch]
        
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


# ------------------------------------------------------------
# 3. JOBS数据集加载函数
# ------------------------------------------------------------
def load_and_format_covariates_jobs(file_path):
    """加载jobs数据集的协变量"""
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, 3:]  # 从第4列开始是协变量 X1-X17
    return x


def load_all_other_crap_jobs(file_path):
    """加载jobs数据集的其他变量"""
    data = np.loadtxt(file_path, delimiter=',')
    t = data[:, 0].reshape(-1, 1)  # 处理T
    y = data[:, 1].reshape(-1, 1)  # 结果Y（二分类0/1）
    e = data[:, 2].reshape(-1, 1)  # 倾向得分e
    return t, y, e


def _extract_dataset_idx(filename):
    """从文件名提取数据集索引号。例如: 'dataset_5.csv' -> 5"""
    import re
    match = re.search(r'dataset_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None


def load_precomputed_true_ate(data_base_dir, dataset_idx):
    """
    从预计算的 .npy 文件加载真实 ATE (概率差)。
    优先加载 true_ate_{dataset_idx}.npy, 否则用 LogisticRegression 兜底估计。
    """
    true_ate_file = os.path.join(data_base_dir, f'true_ate_{dataset_idx}.npy')
    if os.path.exists(true_ate_file):
        true_ate = float(np.load(true_ate_file))
        return true_ate

    # ---- LogisticRegression 兜底估计 ----
    print(f"警告: 未找到预计算的真实ATE ({true_ate_file}), 使用 LogisticRegression 估计")
    try:
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

        X_target = X_orig
        p_y1_given_x_t1 = mu1_model.predict_proba(X_target)[:, 1]
        p_y1_given_x_t0 = mu0_model.predict_proba(X_target)[:, 1]
        true_ate = (p_y1_given_x_t1 - p_y1_given_x_t0).mean()
        print(f"估计的真实ATE: {true_ate:.4f}")
        return true_ate
    except Exception as e:
        print(f"计算真实ATE时出错: {str(e)}")
        return None


# ------------------------------------------------------------
# 4. 评估函数
# ------------------------------------------------------------
def evaluate_ate_binary(model, x, device):
    """评估二分类模型的ATE预测（仅使用神经网络输出）"""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).float().to(device)
        output = model(x_tensor)
        q_t0 = torch.sigmoid(output[:, 0:1]).detach().cpu().numpy()
        q_t1 = torch.sigmoid(output[:, 1:2]).detach().cpu().numpy()
        ate_pred = (q_t1 - q_t0).mean()
    
    return ate_pred


# ------------------------------------------------------------
# 5. 第一阶段：源域预训练
# ------------------------------------------------------------
def train_source_model(x_s, t_s, y_s, model, criterion, optimizer, scheduler, 
                       early_stopper, epochs, batchsize, device, verbose=True):
    """
    在源域数据上训练模型
    
    参数:
        x_s, t_s, y_s: 源域数据
        model: 神经网络模型（TarNet或DragonNet）
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        early_stopper: 早停器
        epochs: 训练轮数
        batchsize: 批次大小
        device: 计算设备
        verbose: 是否输出训练日志
    
    返回:
        trained_model: 训练完成的模型
        best_loss: 最佳验证损失
    """
    print("="*60)
    print("第一阶段：源域预训练")
    print("="*60)
    
    x_s_tensor = torch.from_numpy(x_s).float().to(device)
    t_s_tensor = torch.from_numpy(t_s).float().to(device)
    y_s_tensor = torch.from_numpy(y_s).float().to(device)
    
    dataset = TensorDataset(x_s_tensor, t_s_tensor, y_s_tensor)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, t_batch, y_batch in loader:
            yt_batch = torch.cat([y_batch, t_batch], dim=1)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, yt_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        epoch_loss /= num_batches
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        
        scheduler.step(epoch_loss)
        
        if early_stopper.early_stop(epoch_loss, model):
            if verbose:
                print(f"早停触发于Epoch {epoch}")
            break
        
        if verbose and epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"源域训练 - Epoch {epoch}: Loss={epoch_loss:.4f}, LR={current_lr:.6f}")
    
    early_stopper.load_best_model(model)
    print(f"源域预训练完成 - 最佳损失: {best_loss:.4f}")
    
    return model, best_loss


# ------------------------------------------------------------
# 6. 第二阶段：目标域热启动训练
# ------------------------------------------------------------
def warmstart_target_model(x_t, t_t, y_t, source_model, criterion, optimizer, scheduler,
                           early_stopper, epochs, batchsize, device, verbose=True):
    """
    使用目标域数据进行热启动训练
    
    参数:
        x_t, t_t, y_t: 目标域数据
        source_model: 源域预训练模型（参数将被复制）
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        early_stopper: 早停器
        epochs: 训练轮数
        batchsize: 批次大小
        device: 计算设备
        verbose: 是否输出训练日志
    
    返回:
        target_model: 热启动训练完成的模型
        best_loss: 最佳验证损失
    """
    print("\n" + "="*60)
    print("第二阶段：目标域热启动训练")
    print("="*60)
    
    # 创建目标域模型并复制源域参数
    target_model = type(source_model)(x_t.shape[1]).to(device)
    target_model.load_state_dict(source_model.state_dict())
    
    # 确保优化器与目标模型参数关联
    for param_group in optimizer.param_groups:
        param_group['params'] = list(target_model.parameters())
    
    x_t_tensor = torch.from_numpy(x_t).float().to(device)
    t_t_tensor = torch.from_numpy(t_t).float().to(device)
    y_t_tensor = torch.from_numpy(y_t).float().to(device)
    
    dataset = TensorDataset(x_t_tensor, t_t_tensor, y_t_tensor)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        target_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, t_batch, y_batch in loader:
            yt_batch = torch.cat([y_batch, t_batch], dim=1)
            
            optimizer.zero_grad()
            outputs = target_model(x_batch)
            loss = criterion(outputs, yt_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        epoch_loss /= num_batches
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        
        scheduler.step(epoch_loss)
        
        if early_stopper.early_stop(epoch_loss, target_model):
            if verbose:
                print(f"早停触发于Epoch {epoch}")
            break
        
        if verbose and epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"目标域训练 - Epoch {epoch}: Loss={epoch_loss:.4f}, LR={current_lr:.6f}")
    
    early_stopper.load_best_model(target_model)
    print(f"目标域热启动训练完成 - 最佳损失: {best_loss:.4f}")
    
    return target_model, best_loss


# ------------------------------------------------------------
# 7. 主程序：Warmstart对比迁移模型（JOBS数据集）
# ------------------------------------------------------------
def run_warmstart_baseline_jobs(data_base_dir=r'C:\Users\liruy\Desktop\jobs3',
                                 output_dir='./warmstart_baseline_jobs',
                                 knob='tarnet',
                                 lr_source=5e-4,
                                 lr_target=5e-5,  # 目标域使用较小学习率
                                 l1_reg=0.01,
                                 weight_decay=1e-4,
                                 epochs_source=200,
                                 epochs_target=200,
                                 batchsize=64,
                                 targeted_reg=True,
                                 tarreg_ratio=0.5,
                                 early_stop_patience=20,
                                 save_models=True,
                                 verbose=True):
    """
    运行JOBS数据集的Warmstart对比迁移模型（无对抗适应）
    
    参数:
        data_base_dir: JOBS数据集目录
        output_dir: 输出目录
        knob: 模型类型（'tarnet'或'dragonnet'）
        lr_source: 源域训练学习率
        lr_target: 目标域训练学习率
        l1_reg: L1正则化系数
        weight_decay: 权重衰减
        epochs_source: 源域训练轮数
        epochs_target: 目标域训练轮数
        batchsize: 批次大小
        targeted_reg: 是否使用Targeted Regularization
        tarreg_ratio: Targeted Regularization比例
        early_stop_patience: 早停耐心值
        save_models: 是否保存模型
        verbose: 是否输出详细日志
    
    返回:
        final_output: 包含所有模拟结果的列表
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"选择模型: {knob}（Warmstart对比迁移模型 - JOBS数据集）")
    print(f"源域学习率: {lr_source}, 目标域学习率: {lr_target}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集列表 (自然排序)
    simulation_files = sorted(
        glob.glob("{}/*.csv".format(data_base_dir)),
        key=lambda f: _extract_dataset_idx(f) or 0
    )
    print(f"找到 {len(simulation_files)} 个数据文件")
    
    final_output = []
    all_train_errors = []
    all_test_errors = []
    
    params = {
        'knob': knob,
        'lr_source': lr_source,
        'lr_target': lr_target,
        'l1_reg': l1_reg,
        'weight_decay': weight_decay,
        'epochs_source': epochs_source,
        'epochs_target': epochs_target,
        'batchsize': batchsize,
        'targeted_reg': targeted_reg,
        'tarreg_ratio': tarreg_ratio,
        'adaptation': 'warmstart_only',
        'training_type': 'two_stage',
        'save_models': save_models,
        'data_type': 'binary',
        'test_ratio': 0.3
    }
    
    for idx, simulation_file in enumerate(simulation_files):
        try:
            dataset_idx = _extract_dataset_idx(simulation_file)
            if dataset_idx is None:
                print(f"无法从文件名提取数据集索引: {simulation_file}, 跳过")
                continue

            if verbose:
                print(f"\n{'='*70}")
                print(f"处理文件 {idx+1}/{len(simulation_files)}: "
                      f"{os.path.basename(simulation_file)} (dataset_{dataset_idx})")
                print('='*70)
            
            # 加载JOBS数据
            x = load_and_format_covariates_jobs(simulation_file)
            t, y, e = load_all_other_crap_jobs(simulation_file)
            
            x = x.astype(np.float32)
            t = t.astype(np.float32).reshape(-1, 1)
            y = y.astype(np.float32).reshape(-1, 1)
            
            # 划分源域和目标域
            target_col_idx = 3
            source_idx = np.where(x[:, target_col_idx] == 0)[0]   # 源域
            target_idx = np.where(x[:, target_col_idx] == 1)[0]   # 目标域
            
            if len(source_idx) == 0 or len(target_idx) == 0:
                print(f"跳过空组数据")
                continue
            
            # 提取源域数据（用于第一阶段训练）
            x_s = x[source_idx]
            y_s = y[source_idx]
            t_s = t[source_idx]
            
            # 提取目标域数据
            x_t = x[target_idx]
            y_t = y[target_idx]
            t_t = t[target_idx]
            
            # 从预计算文件加载真实ATE
            true_ate = load_precomputed_true_ate(data_base_dir, dataset_idx)
            if true_ate is None:
                print("无法计算真实ATE，跳过")
                continue

            # ---- 目标域训练集/测试集划分 (7:3 比例) ----
            n_target = len(x_t)
            train_size = int(0.7 * n_target)
            test_size = n_target - train_size
            
            indices = np.random.permutation(n_target)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            
            x_t_train = x_t[train_idx]
            y_t_train = y_t[train_idx]
            t_t_train = t_t[train_idx]
            
            x_t_test = x_t[test_idx]
            y_t_test = y_t[test_idx]
            t_t_test = t_t[test_idx]

            # 目标域ATE是总体统计量, 随机划分后真值不变
            true_ate_train = true_ate
            true_ate_test  = true_ate
            
            print(f"数据划分 - 源域: {x_s.shape}, 目标域: {x_t.shape}")
            print(f"目标域训练集: {x_t_train.shape}, 测试集: {x_t_test.shape}")
            print(f"加载预计算真实ATE (dataset_{dataset_idx}): {true_ate:.4f}")
            
            # 创建损失函数（二分类版本）
            if targeted_reg:
                criterion = make_tarreg_loss_binary(ratio=tarreg_ratio, base_loss=joint_binary_classification_loss)
            else:
                criterion = joint_binary_classification_loss
            
            # ========== 第一阶段：源域预训练 ==========
            input_dim = x_s.shape[1]
            if knob == 'dragonnet':
                source_model = DragonNet(input_dim).to(device)
            elif knob == 'tarnet':
                source_model = TarNet(input_dim).to(device)
            else:
                raise ValueError(f"不支持的模型类型: {knob}")
            
            optimizer_source = optim.Adam(source_model.parameters(), lr=lr_source, weight_decay=weight_decay)
            scheduler_source = optim.lr_scheduler.ReduceLROnPlateau(optimizer_source, mode='min', factor=0.5, patience=10)
            early_stopper_source = EarlyStopper(patience=early_stop_patience, min_delta=0.0001)
            
            source_model, source_best_loss = train_source_model(
                x_s, t_s, y_s, source_model, criterion, optimizer_source, scheduler_source,
                early_stopper_source, epochs_source, batchsize, device, verbose
            )
            
            # 保存源域模型
            if save_models:
                source_model_path = os.path.join(output_dir, f'source_model_{knob}_sim{idx}.pt')
                torch.save(source_model.state_dict(), source_model_path)
                print(f"源域模型保存到: {source_model_path}")
            
            # ========== 第二阶段：目标域热启动训练 (仅在训练集上) ==========
            optimizer_target = optim.Adam(source_model.parameters(), lr=lr_target, weight_decay=weight_decay)
            scheduler_target = optim.lr_scheduler.ReduceLROnPlateau(optimizer_target, mode='min', factor=0.5, patience=10)
            early_stopper_target = EarlyStopper(patience=early_stop_patience, min_delta=0.0001)
            
            target_model, target_best_loss = warmstart_target_model(
                x_t_train, t_t_train, y_t_train,   # 仅在训练集上热启动
                source_model, criterion, optimizer_target, scheduler_target,
                early_stopper_target, epochs_target, batchsize, device, verbose
            )
            
            # 保存目标域模型
            if save_models:
                target_model_path = os.path.join(output_dir, f'target_model_{knob}_sim{idx}.pt')
                torch.save(target_model.state_dict(), target_model_path)
                print(f"目标域模型保存到: {target_model_path}")
            
            # ========== 评估：训练集和测试集分别评估 ==========
            # 源域模型在目标域测试集上的表现（无迁移）
            source_ate_pred_test = evaluate_ate_binary(source_model, x_t_test, device)
            source_ate_error_test = abs(source_ate_pred_test - true_ate_test)
            
            # 热启动后目标域模型 — 训练集
            target_ate_pred_train = evaluate_ate_binary(target_model, x_t_train, device)
            target_ate_error_train = abs(target_ate_pred_train - true_ate_train)
            
            # 热启动后目标域模型 — 测试集
            target_ate_pred_test = evaluate_ate_binary(target_model, x_t_test, device)
            target_ate_error_test = abs(target_ate_pred_test - true_ate_test)
            
            all_train_errors.append(target_ate_error_train)
            all_test_errors.append(target_ate_error_test)
            
            # 记录结果
            result = {
                'sim_idx': idx,
                'file': os.path.basename(simulation_file),
                'dataset_idx': dataset_idx,
                'ate_true': float(true_ate),
                # 源域模型（在测试集上评估）
                'source_ate_pred_test': float(source_ate_pred_test),
                'source_ate_error_test': float(source_ate_error_test),
                # 热启动后 — 训练集
                'target_train_ate_pred': float(target_ate_pred_train),
                'target_train_ate_error': float(target_ate_error_train),
                # 热启动后 — 测试集
                'target_test_ate_pred': float(target_ate_pred_test),
                'target_test_ate_error': float(target_ate_error_test),
                'source_best_loss': float(source_best_loss),
                'target_best_loss': float(target_best_loss),
                'source_domain_size': len(source_idx),
                'target_domain_size': len(target_idx),
                'train_size': len(x_t_train),
                'test_size': len(x_t_test),
                'params': params
            }
            final_output.append(result)
            
            if verbose:
                print("\n" + "="*60)
                print(f"模拟 {idx} 评估结果")
                print("="*60)
                print(f"源域模型 (测试集ATE误差): {source_ate_error_test:.4f}")
                print(f"热启动后 — 训练集ATE误差: {target_ate_error_train:.4f}")
                print(f"热启动后 — 测试集ATE误差: {target_ate_error_test:.4f}")
        
        except Exception as e:
            print(f"处理 {simulation_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总结果
    if all_train_errors and all_test_errors:
        mean_train_error = np.mean(all_train_errors)
        std_train_error = np.std(all_train_errors)
        mean_test_error = np.mean(all_test_errors)
        std_test_error = np.std(all_test_errors)
        
        print(f"\n{'='*70}")
        print(f"{knob.capitalize()} - Warmstart对比迁移模型（无对抗适应）- JOBS数据集")
        print(f"参数配置: {params}")
        print(f"成功运行: {len(all_train_errors)}/{len(simulation_files)}")
        print(f"训练集ATE误差 - 均值: {mean_train_error:.4f}, 标准差: {std_train_error:.4f}")
        print(f"测试集ATE误差 - 均值: {mean_test_error:.4f}, 标准差: {std_test_error:.4f}")
        print('='*70)
    else:
        mean_train_error = std_train_error = np.nan
        mean_test_error = std_test_error = np.nan
    
    summary = {
        'summary': {
            'mean_train_error': float(mean_train_error) if not np.isnan(mean_train_error) else None,
            'std_train_error': float(std_train_error) if not np.isnan(std_train_error) else None,
            'mean_test_error': float(mean_test_error) if not np.isnan(mean_test_error) else None,
            'std_test_error': float(std_test_error) if not np.isnan(std_test_error) else None,
            'successful_runs': len(all_train_errors),
            'params': params
        }
    }
    final_output.append(summary)
    
    # 保存结果
    output_file = os.path.join(output_dir, f'warmstart_baseline_{knob}_jobs_results.json')
    with open(output_file, 'w') as fp:
        json.dump(final_output, fp, indent=2)
    
    print(f"结果保存到: {output_file}")
    return final_output


# ------------------------------------------------------------
# 8. turn_knob接口
# ------------------------------------------------------------
def turn_knob(data_base_dir=r'C:\Users\liruy\Desktop\jobs3',
              knob='tarnet',
              lr_source=5e-4,
              lr_target=5e-5,
              l1_reg=0.01,
              weight_decay=1e-4,
              epochs_source=200,
              epochs_target=200,
              batchsize=64,
              targeted_reg=True,
              tarreg_ratio=0.5,
              early_stop_patience=20,
              save_models=True,
              verbose=True):
    """
    JOBS数据集Warmstart对比迁移模型的turn_knob接口
    """
    print(f"{'='*70}")
    print(f"运行 {knob.capitalize()} - Warmstart对比迁移模型（无对抗适应）- JOBS数据集")
    print(f"{'='*70}")
    
    return run_warmstart_baseline_jobs(
        data_base_dir=data_base_dir,
        knob=knob,
        lr_source=lr_source,
        lr_target=lr_target,
        l1_reg=l1_reg,
        weight_decay=weight_decay,
        epochs_source=epochs_source,
        epochs_target=epochs_target,
        batchsize=batchsize,
        targeted_reg=targeted_reg,
        tarreg_ratio=tarreg_ratio,
        early_stop_patience=early_stop_patience,
        save_models=save_models,
        verbose=verbose
    )


# ------------------------------------------------------------
# 9. 命令行入口
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='JOBS数据集 - Warmstart对比迁移模型（无对抗适应）')
    
    parser.add_argument('--data_base_dir', type=str, default=r'C:\Users\liruy\Desktop\jobs3',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./warmstart_baseline_jobs',
                        help='输出目录路径')
    parser.add_argument('--knob', type=str, default='tarnet', choices=['dragonnet', 'tarnet'])
    parser.add_argument('--lr_source', type=float, default=5e-4, help='源域训练学习率')
    parser.add_argument('--lr_target', type=float, default=5e-5, help='目标域训练学习率')
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1正则化')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--epochs_source', type=int, default=200, help='源域训练轮数')
    parser.add_argument('--epochs_target', type=int, default=200, help='目标域训练轮数')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--targeted_reg', action='store_true', default=True)
    parser.add_argument('--tarreg_ratio', type=float, default=0.5)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--save_models', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    turn_knob(
        data_base_dir=args.data_base_dir,
        knob=args.knob,
        lr_source=args.lr_source,
        lr_target=args.lr_target,
        l1_reg=args.l1_reg,
        weight_decay=args.weight_decay,
        epochs_source=args.epochs_source,
        epochs_target=args.epochs_target,
        batchsize=args.batchsize,
        targeted_reg=args.targeted_reg,
        tarreg_ratio=args.tarreg_ratio,
        early_stop_patience=args.early_stop_patience,
        save_models=args.save_models,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
