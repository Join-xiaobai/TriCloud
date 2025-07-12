from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, input_dim=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)  # 修改为 input_dim^2（6x6=36）
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.input_dim = input_dim  # 添加这一行

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 生成 6x6 的单位矩阵
        iden = torch.eye(self.input_dim).flatten().to(x.device)  # 使用 input_dim
        iden = iden.view(1, self.input_dim * self.input_dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.input_dim, self.input_dim)  # 调整为 [B, 6, 6]
        return x


# STNkd 是一个用于处理kD点云数据的空间变换网络
# 输入点云数据，
# 输出 kxk 的变换矩阵，用于对输入的点云数据进行空间变换。
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        # 1维卷积层，用于从输入的点云数据中提取特征
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 全连接层用于将提取的特征映射到最终的变换矩阵
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        # 激活函数，用于引入非线性
        self.relu = nn.ReLU()
        # 批归一化层，用于加速训练并提高模型的稳定性
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # max将每个点的特征聚合为全局特征。
        # 对张量x在第2个维度上取最大值，并保持输出张量的维度不变： (batch_size, channels, num_points) -》 (batch_size, channels, 1)
        # 具体来说，torch.max函数会返回两个值：最大值和最大值的索引。这里我们只取最大值，因此使用[0]来获取最大值。
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        # 全连接层，将全局特征映射为一个3x3的变换矩阵
        x = self.fc3(x)
        # 将生成的变换矩阵应用到输入的点云数据上，实现空间变换：
        # np.eye(self.k): 生成一个大小为 k x k 的单位矩阵，.flatten(): 将生成的单位矩阵展平成一个一维数组
        # .astype(np.float32): 将数组中的元素转换为 float32 类型，以便与 PyTorch 张量兼容。
        # .view(1, self.k*self.k): 将张量重新调整为形状为 (1, k*k) 的二维张量。
        # .repeat(batchsize, 1): 将张量在第一个维度上重复 batchsize 次，生成形状为 (batchsize, k*k) 的张量。
        # 最终生成的 iden 张量是一个形状为 (b, k*k) 的张量，其中每一行都是一个展平的单位矩阵。这个张量通常用于在神经网络中初始化或调整变换矩阵。
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)  # (b, k*k)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# 是 PointNet 中的一个核心模块，用于从点云数据中提取全局特征或局部特征。它的主要作用是通过多层卷积和全连接层，将输入的点云数据转换为高维特征表示，同时支持空间变换和特征变换。
# 输入：形状为 (batch_size, 3_T_S, num_points) 的点云数据，其中 3_T_S 表示每个点的 (x, y, z) 坐标，num_points 是点的数量。
# 输出：
# - 如果 global_feat 为 True，则输出一个形状为 (batch_size, 1024) 的张量，表示全局特征表示。
# - 如果 global_feat 为 False，则输出一个形状为 (batch_size, 1024, num_points) 的张量，表示每个点的局部特征表示。
# - 如果 feature_transform 为 True，则输出一个形状为 (batch_size, 64, 64) 的变换矩阵，表示局部特征到全局特征的变换关系。
class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_dim=3):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(input_dim=input_dim)
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]  #  = num_points
        trans = self.stn(x) # (batch_size, 3_T_S, 3_T_S)
        x = x.transpose(2, 1) # (batch_size, num_points, 3_T_S)
        # 批量矩阵乘法，用于将变换矩阵应用到点云数据上
        x = torch.bmm(x, trans) # (batch_size, num_points, 3_T_S)

        x = x.transpose(2, 1) # x 形状恢复为 (batch_size, 3_T_S, num_points)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, num_points)  （conv1：3_T_S-》64）

        if self.feature_transform:  # 特征变换（True or False） STNkd对STN3d 处理过的数据 再处理
            trans_feat = self.fstn(x)  #  (batch_size, 64, num_points) -》(batch_size, 64, 64)
            x = x.transpose(2,1) # (batch_size, num_points, 64)
            x = torch.bmm(x, trans_feat)  # (batch_size, num_points, 64)
            x = x.transpose(2,1)  # (batch_size, 64, num_points)
        else:
            trans_feat = None

        pointfeat = x  # (batch_size, 64, num_points)  中间特征
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, num_points)
        x = self.bn3(self.conv3(x))  # (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]  #  (batch_size, 1024, 1)
        x = x.view(-1, 1024) # (batch_size, 1024)  全局特征
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)  # (batch_size, 1024, num_points)
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # # 输出形状为 (batch_size, 1088（1024 + 64）, num_points)


class SelfAttention(nn.Module):
  def __init__(self, in_dim):
      super(SelfAttention, self).__init__()
      self.query = nn.Linear(in_dim, in_dim)
      self.key = nn.Linear(in_dim, in_dim)
      self.value = nn.Linear(in_dim, in_dim)
      self.softmax = nn.Softmax(dim=2)

  def forward(self, x):
      Q = self.query(x)
      K = self.key(x)
      V = self.value(x)
      attention = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5))
      out = torch.bmm(attention, V)
      return out


class PointCloudTransformer(nn.Module):
    def __init__(self, expand_ratio=3, embed_dim=64, num_heads=4):
        super().__init__()
        self.expand_ratio = expand_ratio

        # 展平后的特征处理层
        self.flatten_fc = nn.Linear(1, embed_dim)  # 处理展平后的单通道

        # Transformer配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 维度扩展层
        self.expansion = nn.Sequential(
            nn.Linear(embed_dim, expand_ratio),
            nn.ReLU()
        )

    def forward(self, x):
        # 原始输入: [B, C, N]
        B, C, N = x.shape

        # 展平处理
        x = x.view(B, 1, -1)  # [B, 1, C*N]
        x = x.permute(0, 2, 1)  # [B, C*N, 1]

        # 特征嵌入
        x = self.flatten_fc(x)  # [B, C*N, D]

        # Transformer处理
        x = self.transformer(x)  # [B, C*N, D]

        # 维度扩展
        x = self.expansion(x)  # [B, C*N, 3_T_S]
        x = x.view(B, C, -1)  # [B, C, N*3_T_S]

        return x


# PointNetCls 是 PointNet 中用于分类任务的模块
# 通过 PointNetfeat 提取点云数据的全局特征，然后通过全连接层进行分类。
class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False, input_dim=3):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, input_dim=input_dim)
        # self.attention = SelfAttention(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.transformer = PointCloudTransformer()  # 根据实际点数修改

#  926151
    def forward(self, x):
        # x：全局特征，形状为 (batch_size, 1024)。
        # trans：空间变换矩阵，形状为 (batch_size, 3_T_S, 3_T_S)。
        # trans_feat：特征变换矩阵，形状为 (batch_size, 64, 64)（如果 feature_transform=True）。
        x = self.transformer(x)

        x, trans, trans_feat = self.feat(x)
        # x = self.attention(x.unsqueeze(1)).squeeze(1)  # 应用注意力机制
        x = F.relu(self.bn1(self.fc1(x)))  # (batch_size, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))  # (batch_size, 256)
        x = self.fc3(x) # (batch_size, k)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
