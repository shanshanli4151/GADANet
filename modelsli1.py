#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import numpy as np
import random


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    
    # 全面的数值稳定性处理
    # 1. 检查是否有 NaN 或 Inf，替换为安全值
    t_pred = torch.nan_to_num(t_pred, nan=0.5, posinf=0.99, neginf=0.01)
    # 2. 严格的裁剪
    t_pred = torch.clamp(t_pred, 1e-7, 1 - 1e-7)
    # 3. 再次检查并修复
    t_pred = torch.nan_to_num(t_pred, nan=0.5, posinf=0.99, neginf=0.01)
    
    try:
        loss = torch.sum(F.binary_cross_entropy(t_pred, t_true, reduction='sum'))
    except:
        # 如果还有异常，使用更安全的替代计算
        t_pred_safe = torch.clamp(t_pred, 1e-5, 1 - 1e-5)
        loss = torch.sum(F.binary_cross_entropy(t_pred_safe, t_true, reduction='sum'))
    
    return loss


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 1]
    # 对logits进行数值稳定性处理
    t_pred = torch.clamp(t_pred, -100, 100)
    return torch.sum(F.binary_cross_entropy_with_logits(t_pred, t_true))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_pred, concat_true):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def transfer_loss(concat_pred, concat_true, params, l1_reg):
    return dragonnet_loss_binarycross(concat_pred, concat_true) + l1_reg * torch.abs(params.view()).sum()


class EpsilonLayer(nn.Module):
    def __init__(self):
        super(EpsilonLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, 1))
        nn.init.normal_(self.weights, mean=0, std=0.05)

    def forward(self, inputs):
        return torch.mm(torch.ones_like(inputs)[:, 0:1], self.weights.T)


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    """
    Create the targeted regularization loss criterion
    Args:
        ratio: Ratio of targeted regularization to use
        dragonnet_loss: Simple loss
    """
    def tarreg_ATE_unbounded_domain_loss(concat_pred, concat_true):
        vanilla_loss = dragonnet_loss(concat_pred, concat_true)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]
        epsilons = concat_pred[:, 3]
        
        # 全面的数值稳定性处理
        # 处理 t_pred
        t_pred = torch.nan_to_num(t_pred, nan=0.5, posinf=0.99, neginf=0.01)
        t_pred = torch.clamp(t_pred, 1e-7, 1 - 1e-7)
        t_pred = torch.nan_to_num(t_pred, nan=0.5, posinf=0.99, neginf=0.01)
        t_pred = (t_pred + 0.01) / 1.02
        t_pred = torch.clamp(t_pred, 1e-7, 1 - 1e-7)
        t_pred = torch.nan_to_num(t_pred, nan=0.5, posinf=0.99, neginf=0.01)
        
        # 处理 epsilons
        epsilons = torch.nan_to_num(epsilons, nan=0.0, posinf=5.0, neginf=-5.0)
        epsilons = torch.clamp(epsilons, -10, 10)
        
        # 处理 y_pred 相关的变量
        y0_pred = torch.nan_to_num(y0_pred, nan=0.0, posinf=100.0, neginf=-100.0)
        y1_pred = torch.nan_to_num(y1_pred, nan=0.0, posinf=100.0, neginf=-100.0)
        
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
        
        # 安全地计算 h，避免除零
        t_pred_safe = torch.clamp(t_pred, 1e-5, 1 - 1e-5)
        h = t_true / t_pred_safe - (1 - t_true) / (1 - t_pred_safe)
        h = torch.nan_to_num(h, nan=0.0, posinf=1000.0, neginf=-1000.0)
        h = torch.clamp(h, -1000, 1000)

        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))

        return vanilla_loss + ratio * targeted_regularization

    return tarreg_ATE_unbounded_domain_loss


def weights_init_normal(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.normal_(params.weight, mean=0.0, std=0.1)  # 减小初始方差
        torch.nn.init.zeros_(params.bias)


def weights_init_zero(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.zeros_(params.weight)
        torch.nn.init.zeros_(params.bias)


def weights_init_uniform(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.uniform_(params.weight, a=-0.1, b=0.1)  # 更保守的初始化
        torch.nn.init.zeros_(params.bias)


class DragonNet(nn.Module):
    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(DragonNet, self).__init__()

        i = 0
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
    
        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU()
        )

        self.t_predictions = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[2]),
                                           nn.Sigmoid())

        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.epsilon = EpsilonLayer()
    
    def init_params(self, std=0.1):
        i = 0
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
    
        self.representation_block.apply(weights_init_normal)
        self.t_predictions.apply(weights_init_uniform)
        self.t0_head.apply(weights_init_uniform)
        self.t1_head.apply(weights_init_uniform)
    
    def forward(self, x):
        # 输入检查
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        x = self.representation_block(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)

        propensity_head = self.t_predictions(x)
        propensity_head = torch.nan_to_num(propensity_head, nan=0.5, posinf=0.99, neginf=0.01)
        
        epsilons = self.epsilon(propensity_head)
        epsilons = torch.nan_to_num(epsilons, nan=0.0, posinf=5.0, neginf=-5.0)

        t0_out = self.t0_head(x)
        t0_out = torch.nan_to_num(t0_out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        t1_out = self.t1_head(x)
        t1_out = torch.nan_to_num(t1_out, nan=0.0, posinf=100.0, neginf=-100.0)

        return torch.cat((t0_out, t1_out, propensity_head, epsilons), 1)


class DragonNet_transfer(DragonNet):
    def __init__(self, in_features, parm, out_features=[200, 100, 1]):
        self.parm = parm
        # 确保 parm 没有 NaN 值
        for key in self.parm:
            self.parm[key] = torch.nan_to_num(self.parm[key], nan=0.0, posinf=0.1, neginf=-0.1)
        super().__init__(in_features, out_features)
        self.init_params()
    
    def init_params(self):
        self.representation_block.apply(weights_init_zero)
        self.t_predictions.apply(weights_init_zero)
        self.t0_head.apply(weights_init_zero)
        self.t1_head.apply(weights_init_zero)
    
    def forward(self, x):
        # 输入检查
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        orig_weight = {}
        for name, param in self.representation_block.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['representation_block.' + name]

        x = self.representation_block(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        for name, param in self.representation_block.named_parameters():
            param.data = orig_weight[name].data
        
        orig_weight = {}
        for name, param in self.t_predictions.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['t_predictions.' + name]
                
        propensity_head = self.t_predictions(x)
        propensity_head = torch.nan_to_num(propensity_head, nan=0.5, posinf=0.99, neginf=0.01)
        
        for name, param in self.t_predictions.named_parameters():
            param.data = orig_weight[name].data
            
        epsilons = self.epsilon(propensity_head)
        epsilons = torch.nan_to_num(epsilons, nan=0.0, posinf=5.0, neginf=-5.0)
        
        orig_weight = {}
        for name, param in self.t0_head.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['t0_head.' + name]
                
        t0_out = self.t0_head(x)
        t0_out = torch.nan_to_num(t0_out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        for name, param in self.t0_head.named_parameters():
            param.data = orig_weight[name].data
        
        orig_weight = {}
        for name, param in self.t1_head.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['t1_head.' + name]
                
        t1_out = self.t1_head(x)
        t1_out = torch.nan_to_num(t1_out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        for name, param in self.t1_head.named_parameters():
            param.data = orig_weight[name].data

        return torch.cat((t0_out, t1_out, propensity_head, epsilons), 1)


class TarNet(nn.Module):
    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(TarNet, self).__init__()
        self.out_features = out_features
        i = 0
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
    
        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU()
        )

        self.t_predictions = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features[2]),
                                           nn.Sigmoid())

        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.epsilon = EpsilonLayer()

    def init_params(self, std=0.1):
        i = 0
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
        self.representation_block.apply(weights_init_normal)
        self.t_predictions.apply(weights_init_uniform)
        self.t0_head.apply(weights_init_uniform)
        self.t1_head.apply(weights_init_uniform)

    def forward(self, x):
        # 输入检查
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        rep_block = self.representation_block(x)
        rep_block = torch.nan_to_num(rep_block, nan=0.0, posinf=100.0, neginf=-100.0)
        
        propensity_head = self.t_predictions(x)
        propensity_head = torch.nan_to_num(propensity_head, nan=0.5, posinf=0.99, neginf=0.01)
        
        epsilons = self.epsilon(propensity_head)
        epsilons = torch.nan_to_num(epsilons, nan=0.0, posinf=5.0, neginf=-5.0)
        
        t0_out = self.t0_head(rep_block)
        t0_out = torch.nan_to_num(t0_out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        t1_out = self.t1_head(rep_block)
        t1_out = torch.nan_to_num(t1_out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return torch.cat((t0_out, t1_out, propensity_head, epsilons), 1)


class TarNet_transfer(TarNet):
    def __init__(self, in_features, parm, out_features=[200, 100, 1]):
        self.parm = parm
        # 确保 parm 没有 NaN 值
        for key in self.parm:
            self.parm[key] = torch.nan_to_num(self.parm[key], nan=0.0, posinf=0.1, neginf=-0.1)
        super().__init__(in_features, out_features)
        self.init_params()
    
    def init_params(self):
        self.representation_block.apply(weights_init_zero)
        self.t_predictions.apply(weights_init_zero)
        self.t0_head.apply(weights_init_zero)
        self.t1_head.apply(weights_init_zero)
    
    def forward(self, x):
        # 输入检查
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        orig_weight = {}
        for name, param in self.representation_block.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['representation_block.' + name]

        rep_block = self.representation_block(x)
        rep_block = torch.nan_to_num(rep_block, nan=0.0, posinf=100.0, neginf=-100.0)
        
        for name, param in self.representation_block.named_parameters():
            param.data = orig_weight[name].data
            
        orig_weight = {}
        for name, param in self.t_predictions.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['t_predictions.' + name]
                
        propensity_head = self.t_predictions(x)
        propensity_head = torch.nan_to_num(propensity_head, nan=0.5, posinf=0.99, neginf=0.01)
        
        for name, param in self.t_predictions.named_parameters():
            param.data = orig_weight[name].data
        
        epsilons = self.epsilon(propensity_head)
        epsilons = torch.nan_to_num(epsilons, nan=0.0, posinf=5.0, neginf=-5.0)
        
        orig_weight = {}
        for name, param in self.t0_head.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['t0_head.' + name]
                
        t0_out = self.t0_head(rep_block)
        t0_out = torch.nan_to_num(t0_out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        for name, param in self.t0_head.named_parameters():
            param.data = orig_weight[name].data
        
        orig_weight = {}
        for name, param in self.t1_head.named_parameters():
            orig_weight.update({name: param.clone()})
            param.data = param.data + self.parm['t1_head.' + name]
                
        t1_out = self.t1_head(rep_block)
        t1_out = torch.nan_to_num(t1_out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        for name, param in self.t1_head.named_parameters():
            param.data = orig_weight[name].data

        return torch.cat((t0_out, t1_out, propensity_head, epsilons), 1)