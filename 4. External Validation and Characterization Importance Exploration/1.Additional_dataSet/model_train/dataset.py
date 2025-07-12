import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch

def pc_normalize(pc):
    """归一化点云"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m if m != 0 else pc  # 避免除以零
    return pc


class BinaryLabelDataLoader(Dataset):
    def __init__(self, file_0, file_1, normal_channel=True, split='train', test_size=0.2, seed=None, groupNum=3):
        self.normal_channel = normal_channel
        self.split = split
        self.groupNum = groupNum

        # 解决方案1: 使用skiprows跳过表头
        # 加载标签为0的数据
        data_0 = np.loadtxt(file_0, delimiter='\t', skiprows=1).astype(np.float32)
        random_data = np.random.rand(data_0.shape[0], 64).astype(np.float32)  # 生成0-1随机数
        data_0[:, 3:] = random_data  # 从第3列开始替换（保留前3列）
        label_0 = np.zeros((len(data_0), 1), dtype=np.int32)

        # 加载标签为1的数据
        data_1 = np.loadtxt(file_1, delimiter='\t', skiprows=1).astype(np.float32)
        random_data = np.random.rand(data_1.shape[0], 64).astype(np.float32)  # 生成0-1随机数
        data_1[:, 3:] = random_data  # 从第3列开始替换（保留前3列）
        label_1 = np.ones((len(data_1), 1), dtype=np.int32)

        # 合并数据和标签
        data = np.concatenate([data_0, data_1], axis=0)
        labels = np.concatenate([label_0, label_1], axis=0)

        # 划分数据集
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=test_size, random_state=seed, stratify=labels)

        if split == 'train':
            self.data = train_data
            self.labels = train_labels
        elif split == 'test':
            self.data = test_data
            self.labels = test_labels

        # 归一化整个点云
        self.data[:, :3] = pc_normalize(self.data[:, :3])

    def __len__(self):
        return len(self.data)  # 返回点的数量

    def __getitem__(self, index):
        point = self.data[index]  # shape: (67,) 前三列是XYZ，后64列是特征
        label = self.labels[index]  # 标签

        # 提取XYZ坐标 (前三列)
        xyz = point[:3]

        # 提取64维特征并分成4组16维
        features = point[3:]  # shape: (64,)
        grouped_features = np.reshape(features, (self.groupNum, -1))  # 分成4组16维特征

        # 将XYZ与每组16维特征拼接，形成4个19维向量
        processed_points = []
        for group in grouped_features:
            combined = np.concatenate([xyz, group])  # shape: (19,)
            processed_points.append(combined)

        # 转换为(19,4)张量
        processed_points = np.array(processed_points).T  # shape: (19,4)

        # 是否保留法向量 (这里假设前三列是XYZ，没有法向量信息)
        if not self.normal_channel:
            processed_points = processed_points[:3]  # 只保留XYZ坐标部分

        # 转换为PyTorch张量
        processed_points = torch.from_numpy(processed_points).float()
        label = torch.from_numpy(label).long()

        return processed_points, label


if __name__ == '__main__':
    import torch
    
    # 设置你的两个txt文件的路径
    file_0 = './data/drug_gene_disease_features_positive_mapped.tsv'
    file_1 = './data/drug_gene_disease_features_negative_mapped.tsv'

    # 创建训练集和测试集的数据加载器
    train_data = BinaryLabelDataLoader(file_0=file_0, file_1=file_1,
                                       normal_channel=True, split='train', test_size=0.2, seed=42)
    test_data = BinaryLabelDataLoader(file_0=file_0, file_1=file_1,
                                      normal_channel=True, split='test', test_size=0.2, seed=42)

    train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=12, shuffle=False)

    # 打印训练集和测试集的第一个batch的形状
    for points, labels in train_loader:
        print("Train Batch:")
        print(points.shape)  # 应该是 [batch_size, num_features] 其中 num_features 是6（包括xyz和法向量）或3（只有xyz）
        print(labels.shape)  # 应该是 [batch_size, additional_dataSet]
        break

    for points, labels in test_loader:
        print("Test Batch:")
        print(points.shape)  # 应该是 [batch_size, num_features] 其中 num_features 是6（包括xyz和法向量）或3（只有xyz）
        print(labels.shape)  # 应该是 [batch_size, additional_dataSet]
        break


    for points, labels in train_loader:
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import seaborn as sns

        # ========================
        # Step 2: 类别分布统计
        # ========================
        
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        print("\n【类别分布】")
        for cls, count in class_distribution.items():
            print(f"Class {cls}: {count} samples")