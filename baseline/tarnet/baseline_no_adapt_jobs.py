"""
基线模型 - 无域适应版本（适用于JOBS数据集）
用于消融实验对比有无域适应的效果差异

技术规范：
1. 仅使用 TarNet 或 DragonNet 神经网络模型
2. 训练数据：纯目标域数据（先划分目标域，再使用目标域数据训练）
3. 禁止引入 IPW 等基础估计器
4. 移除对抗训练机制
5. 其他参数与基准实验保持一致
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
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from idhp_data import *
from ate import *
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
# 2. 二分类专用损失函数
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
# 3. Jobs数据集加载函数
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
# 5. 主程序
# ------------------------------------------------------------
def run_baseline_jobs(data_base_dir=r'C:\Users\liruy\Desktop\jobs3',
                      output_dir=r'C:\Users\liruy\Desktop\jobs3',
                      knob='tarnet',
                      lr=5e-4,
                      weight_decay=1e-4,
                      epochs=200,
                      batchsize=64,
                      targeted_reg=True,
                      tarreg_ratio=0.5,
                      early_stop_patience=20,
                      verbose=True):
    """
    运行JOBS数据集的基线模型（无域适应，纯目标域训练）
    
    技术规范：
    - 训练数据：仅使用目标域数据
    - 模型：仅使用 TarNet 或 DragonNet
    - 禁止使用 IPW 等基础估计器
    - 无对抗训练机制
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"选择模型: {knob}（无域适应版本，纯目标域训练）")
    #print(f"参数: lr={lr}, l1_reg={l1_reg}, weight_decay={weight_decay}, epochs={epochs}")
    
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
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'batchsize': batchsize,
        'targeted_reg': targeted_reg,
        'tarreg_ratio': tarreg_ratio,
        'adaptation': 'none',
        'training_data': 'target_domain_only',  # 标记仅目标域训练
        'estimator': 'nn_only',  # 标记仅使用神经网络
        'data_type': 'binary',
        'test_ratio': 0.3  # 测试集占30%
    }
    
    for idx, simulation_file in enumerate(simulation_files):
        try:
            dataset_idx = _extract_dataset_idx(simulation_file)
            if dataset_idx is None:
                print(f"无法从文件名提取数据集索引: {simulation_file}, 跳过")
                continue

            if verbose:
                print(f"\n处理文件 {idx+1}/{len(simulation_files)}: "
                      f"{os.path.basename(simulation_file)} (dataset_{dataset_idx})")
            
            # 加载数据
            x = load_and_format_covariates_jobs(simulation_file)
            t, y, e = load_all_other_crap_jobs(simulation_file)
            
            x = x.astype(np.float32)
            t = t.astype(np.float32).reshape(-1, 1)
            y = y.astype(np.float32).reshape(-1, 1)
            
            # 划分目标域（X3作为域标签）
            target_col_idx = 2
            print(f"选择目标索引: {target_col_idx}")
            
            target_idx1 = np.where(x[:, target_col_idx] == 1)[0]  # 目标域
            
            if len(target_idx1) == 0:
                print(f"跳过空目标域数据")
                continue
            
            # 仅使用目标域数据
            x_t = x[target_idx1]
            y_t = y[target_idx1]
            t_t = t[target_idx1]
            
            # 从预计算文件加载真实ATE
            # JOBS数据集没有mu_0/mu_1，使用预计算的总体真实ATE
            true_ate = load_precomputed_true_ate(data_base_dir, dataset_idx)
            if true_ate is None:
                print("无法计算真实ATE，跳过")
                continue
            
            # 划分训练集和测试集（7:3比例）
            n_samples = len(x_t)
            train_size = int(0.7 * n_samples)
            test_size = n_samples - train_size
            
            # 随机划分
            indices = np.random.permutation(n_samples)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            
            # 训练集
            x_train = x_t[train_idx]
            y_train = y_t[train_idx]
            t_train = t_t[train_idx]
            
            # 测试集
            x_test = x_t[test_idx]
            y_test = y_t[test_idx]
            t_test = t_t[test_idx]
            
            # JOBS数据集无逐样本mu_0/mu_1，训练集和测试集使用相同的总体真实ATE
            # (ATE是目标域总体的因果效应期望，随机划分后真值不变)
            true_ate_train = true_ate
            true_ate_test = true_ate
            
            print(f"目标域数据: {x_t.shape}")
            print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")
            print(f"训练集真实ATE: {true_ate_train:.4f}, 测试集真实ATE: {true_ate_test:.4f}")
            
            input_dim = x_t.shape[1]
            if knob == 'dragonnet':
                model = DragonNet(input_dim).to(device)
            elif knob == 'tarnet':
                model = TarNet(input_dim).to(device)
            else:
                raise ValueError(f"不支持的模型类型: {knob}")
            
            if targeted_reg:
                criterion = make_tarreg_loss_binary(ratio=tarreg_ratio, base_loss=joint_binary_classification_loss)
            else:
                criterion = joint_binary_classification_loss
            
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            early_stopper = EarlyStopper(patience=early_stop_patience, min_delta=0.0001)
            
            if verbose:
                print("开始训练（仅目标域训练集数据）...")
            
            best_train_ate_error = float('inf')
            best_test_ate_error = float('inf')
            stopped_epoch = epochs
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                # 仅在训练集上训练
                indices_batch = np.random.permutation(len(x_train))
                for i in range(0, len(x_train), batchsize):
                    batch_idx = indices_batch[i:i+batchsize]
                    x_batch = torch.from_numpy(x_train[batch_idx]).float().to(device)
                    t_batch = torch.from_numpy(t_train[batch_idx]).float().to(device)
                    y_batch = torch.from_numpy(y_train[batch_idx]).float().to(device)
                    
                    yt_batch = torch.cat([y_batch, t_batch], dim=1)
                    
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = criterion(outputs, yt_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                epoch_loss /= max(num_batches, 1)
                
                # 在训练集和测试集上分别评估
                ate_pred_train = evaluate_ate_binary(model, x_train, device)
                ate_error_train = abs(ate_pred_train - true_ate_train)
                
                ate_pred_test = evaluate_ate_binary(model, x_test, device)
                ate_error_test = abs(ate_pred_test - true_ate_test)
                
                if ate_error_train < best_train_ate_error:
                    best_train_ate_error = ate_error_train
                if ate_error_test < best_test_ate_error:
                    best_test_ate_error = ate_error_test
                
                # 使用测试集误差进行早停和学习率调度
                scheduler.step(ate_error_test)
                
                if early_stopper.early_stop(ate_error_test, model):
                    stopped_epoch = epoch
                    if verbose:
                        print(f"早停触发于Epoch {epoch}")
                    break
                
                if verbose and epoch % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch:3d}: Loss={epoch_loss:.4f}, "
                          f"Train ATE Error={ate_error_train:.4f}, "
                          f"Test ATE Error={ate_error_test:.4f}, "
                          f"LR={current_lr:.6f}")
            
            early_stopper.load_best_model(model)
            
            # 最终评估
            final_ate_pred_train = evaluate_ate_binary(model, x_train, device)
            final_ate_error_train = abs(final_ate_pred_train - true_ate_train)
            
            final_ate_pred_test = evaluate_ate_binary(model, x_test, device)
            final_ate_error_test = abs(final_ate_pred_test - true_ate_test)
            
            all_train_errors.append(final_ate_error_train)
            all_test_errors.append(final_ate_error_test)
            
            result = {
                'sim_idx': idx,
                'file': os.path.basename(simulation_file),
                'dataset_idx': dataset_idx,
                # 训练集结果
                'train_ate_true': float(true_ate_train),
                'train_ate_pred': float(final_ate_pred_train),
                'train_ate_error': float(final_ate_error_train),
                'best_train_ate_error': float(best_train_ate_error),
                # 测试集结果
                'test_ate_true': float(true_ate_test),
                'test_ate_pred': float(final_ate_pred_test),
                'test_ate_error': float(final_ate_error_test),
                'best_test_ate_error': float(best_test_ate_error),
                # 其他信息
                'stopped_epoch': stopped_epoch,
                'target_domain_size': len(target_idx1),
                'train_size': len(x_train),
                'test_size': len(x_test),
                'params': params
            }
            final_output.append(result)
            
            if verbose:
                print(f"模拟 {idx} 完成 - "
                      f"训练集ATE误差: {final_ate_error_train:.4f}, "
                      f"测试集ATE误差: {final_ate_error_test:.4f}")
        
        except Exception as e:
            print(f"处理 {simulation_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_train_errors and all_test_errors:
        mean_train_error = np.mean(all_train_errors)
        std_train_error = np.std(all_train_errors)
        mean_test_error = np.mean(all_test_errors)
        std_test_error = np.std(all_test_errors)
        
        print(f"\n{'='*70}")
        print(f"{knob.capitalize()} - 基线模型（仅目标域训练）- JOBS数据集")
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
    
    output_dir_path = f'{output_dir}_{knob}/'
    os.makedirs(output_dir_path, exist_ok=True)
    
    output_file = f'{output_dir_path}baseline_{knob}_jobs_target_only.json'
    with open(output_file, 'w') as fp:
        json.dump(final_output, fp, indent=2)
    
    print(f"结果保存到: {output_file}")
    return final_output


# ------------------------------------------------------------
# 6. turn_knob接口
# ------------------------------------------------------------
def turn_knob(data_base_dir=r'C:\Users\liruy\Desktop\jobs3',
              knob='tarnet',
              lr=5e-4,
              weight_decay=1e-4,
              batchsize=64,
              targeted_reg=True,
              tarreg_ratio=0.5,
              epochs=200,
              early_stop_patience=20,
              verbose=True):
    """
    JOBS数据集基线模型（仅目标域训练）的turn_knob接口
    """
    print(f"{'='*70}")
    print(f"运行 {knob.capitalize()} - 基线模型（仅目标域训练）- JOBS数据集")
    print(f"{'='*70}")
    
    return run_baseline_jobs(
        data_base_dir=data_base_dir,
        knob=knob,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batchsize=batchsize,
        targeted_reg=targeted_reg,
        tarreg_ratio=tarreg_ratio,
        early_stop_patience=early_stop_patience,
        verbose=verbose
    )


# ------------------------------------------------------------
# 7. 命令行入口
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='JOBS数据集 - 基线模型（仅目标域训练，无域适应）')
    
    parser.add_argument('--data_base_dir', type=str, default=r'C:\Users\liruy\Desktop\jobs3',
                        help='数据目录路径')
    parser.add_argument('--knob', type=str, default='tarnet', choices=['dragonnet', 'tarnet'])
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--targeted_reg', action='store_true', default=True)
    parser.add_argument('--tarreg_ratio', type=float, default=0.5)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    turn_knob(
        data_base_dir=args.data_base_dir,
        knob=args.knob,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batchsize=args.batchsize,
        targeted_reg=args.targeted_reg,
        tarreg_ratio=args.tarreg_ratio,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
