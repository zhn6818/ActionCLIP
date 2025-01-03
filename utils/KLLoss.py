# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.nn.functional as F
import torch.nn as nn

class KLLoss(nn.Module):
    """KL散度损失函数类
    
    该类实现了基于KL散度的损失计算。KL散度用于衡量两个概率分布之间的差异。
    在这里用于计算模型预测分布与目标分布之间的差异。
    
    参数:
        error_metric: 基础损失函数,默认使用KLDivLoss
                     size_average=True表示对batch求平均
                     reduce=True表示对所有元素求和
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        # 初始化KL散度损失函数
        self.error_metric = error_metric

    def forward(self, prediction, label):
        # 获取batch大小
        batch_size = prediction.shape[0]
        
        # 对预测结果进行log_softmax,得到对数概率分布
        # dim=1表示在特征维度上进行softmax
        probs1 = F.log_softmax(prediction, 1)
        
        # 对标签乘以温度系数10后进行softmax,得到目标概率分布
        # 温度系数用于调整分布的平滑程度
        probs2 = F.softmax(label * 10, 1)
        
        # 计算KL散度损失,并乘以batch_size
        # 乘以batch_size是为了得到未归一化的损失
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss