from torch.utils.data import Dataset
import numpy as np
import torch

# 固定随机种子
def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


class CustomDataset(Dataset):
    def __init__(self, pos_file, neg_file, split='train', train_ratio=0.8,
                 val_ratio=0.1, test_ratio=0.1, repeat=3):
        self.samples = []
        self.split = split

        # 加载正样本
        self._load_samples(pos_file, repeat)

        # 加载负样本
        self._load_samples(neg_file, repeat)

        # 划分数据集
        self._split_dataset(train_ratio, val_ratio, test_ratio, split)

    def _load_samples(self, file_path, repeat=3):
        data = np.loadtxt(file_path, delimiter=' ', ndmin=2)  # 确保二维数组

        for row in data:
            if len(row) != 7:
                raise ValueError(f"文件 {file_path} 中存在无效行: 应有7列，实际{len(row)}列")

            # 提取特征和标签
            features = row[:-1]  # 前6列是特征
            label = int(row[-1])  # 最后一列是标签

            # 将特征复制成三行
            # print(features.shape)
            features_repeated = np.tile(features, (repeat, 1))  # 复制成6行
            # print(features_repeated.shape)

            # 将6个特征重塑为 (2, 3_T_S) 形状的点云
            # 2个点，每个点3个坐标值
            point_cloud = features_repeated.reshape(-1, 3)  # 自动计算为(2,3_T_S)
            # print(point_cloud.shape)

            self.samples.append((point_cloud, label))

    def _split_dataset(self, train_ratio, val_ratio, test_ratio, split):
        total_samples = len(self.samples)
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        # 随机划分数据集
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # 根据 split 参数选择数据集
        if split == 'train':
            self.samples = [self.samples[i] for i in train_indices]
        elif split == 'val':
            self.samples = [self.samples[i] for i in val_indices]
        elif split == 'test':
            self.samples = [self.samples[i] for i in test_indices]
        else:
            raise ValueError("split 参数必须是 'train', 'val' 或 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        point_cloud, label = self.samples[idx]
        # 转置点云数据，从 [363, 3_T_S] 变为 [3_T_S, 363]
        point_cloud = point_cloud.transpose(1, 0)
        return torch.tensor(point_cloud).float(), torch.tensor(label).long()


# 示例用法
if __name__ == '__main__':
    pos_file = '../dataset/new_data/data_1.txt'
    neg_file = '../dataset/new_data/data_0.txt'

    # 创建数据集
    train_dataset = CustomDataset(pos_file, neg_file, split='train')
    val_dataset = CustomDataset(pos_file, neg_file, split='val')
    test_dataset = CustomDataset(pos_file, neg_file, split='test')

    print(f"训练集样本数量: {len(train_dataset)}")
    print(f"验证集样本数量: {len(val_dataset)}")
    print(f"测试集样本数量: {len(test_dataset)}")

    # 获取第一个样本
    point_cloud, label = train_dataset[0]
    print(f"训练集点云数据形状: {point_cloud.shape}")  # 输出: torch.Size([363, 3_T_S])
    print(f"训练集标签: {label}")  # 输出: 0 或 1
