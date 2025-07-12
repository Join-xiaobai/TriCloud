import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, input_dim=3):
        """
        空间变换网络（Spatial Transformer Network），用于学习输入点云数据的空间变换矩阵。

        参数:
        - input_dim: 输入特征维度，默认为3（即x, y, z坐标）。
        """
        super(STN3d, self).__init__()
        # 卷积层定义
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)  # 输出维度为input_dim^2，用于生成变换矩阵

        # 批归一化层
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.input_dim = input_dim  # 记录输入维度

    def forward(self, x):
        batchsize = x.size()[0]
        # 前向传播过程：卷积 + ReLU + BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]  # 全局最大池化
        x = x.view(-1, 1024)  # 展平

        # 全连接层
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 初始化单位矩阵并添加到输出中，确保变换矩阵接近单位矩阵
        iden = torch.eye(self.input_dim).flatten().to(x.device)
        iden = iden.view(1, self.input_dim * self.input_dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.input_dim, self.input_dim)  # 转换为变换矩阵形式
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_dim=3):
        """
        PointNet 特征提取模块，支持全局特征和局部特征提取。

        参数:
        - global_feat: 是否提取全局特征。
        - feature_transform: 是否应用特征变换。
        - input_dim: 输入特征维度。
        """
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(input_dim=input_dim)  # 空间变换网络
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STN3d(input_dim=64)  # 针对特征的变换网络

    def forward(self, x):
        n_pts = x.size()[2]  # 获取点的数量
        trans = self.stn(x)  # 应用空间变换
        x = x.transpose(2, 1)  # 转置以适应矩阵乘法要求
        x = torch.bmm(x, trans)  # 应用变换
        x = x.transpose(2, 1)  # 恢复原始形状

        x = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积
        if self.feature_transform:
            trans_feat = self.fstn(x)  # 如果需要特征变换，则应用
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x  # 中间特征保存
        x = F.relu(self.bn2(self.conv2(x)))  # 第二层卷积
        x = self.bn3(self.conv3(x))  # 第三层卷积
        x = torch.max(x, 2, keepdim=True)[0]  # 全局最大池化
        x = x.view(-1, 1024)  # 展平

        if self.global_feat:
            return x, trans, trans_feat  # 返回全局特征和变换矩阵
        else:
            x = x.view(-1, 1024, 1).groupNum(1, 1, n_pts)  # 复制全局特征至每个点
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # 连接全局和局部特征


class PointNetCls(nn.Module):
    def __init__(self, class_num=2, feature_transform=False, input_dim=3):
        """
        PointNet 分类器模块。

        参数:
        - class_num: 类别数量。
        - feature_transform: 是否使用特征变换。
        - input_dim: 输入特征维度。
        """
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, input_dim=input_dim)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, class_num)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)  # 提取特征
        x = F.relu(self.bn1(self.fc1(x)))  # 全连接层+ReLU激活
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))  # Dropout+全连接层+ReLU激活
        x = self.fc3(x)  # 输出层
        return F.log_softmax(x, dim=1), trans, trans_feat  # 返回分类结果和变换矩阵


def feature_transform_regularizer(trans):
    """
    特征变换正则化损失函数，鼓励变换矩阵接近单位矩阵。

    参数:
    - trans: 变换矩阵。
    """
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def main():
    model = PointNetCls(input_dim=6)  # 构建模型实例
    print("Model built successfully.")

    points = torch.rand(12, 6, 3)  # 模拟输入数据
    labels = torch.randint(0, 2, (12,))  # 生成随机标签
    print("Labels shape:", labels.shape)
    print("\nInput Points Shape:", points.shape)

    trans = STN3d(input_dim=6)  # 创建STN实例
    out = trans(points)
    print('STN output shape:', out.size())
    print('Feature transform regularizer loss:', feature_transform_regularizer(out))

    with torch.no_grad():  # 测试前向传播
        outputs, _, _ = model(points)
        print("Model Output Shape:", outputs.size())

    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    loss = criterion(outputs, labels)  # 计算损失
    print("Loss value:", loss.item())
    print("✅ Model test passed!")


if __name__ == '__main__':
    main()