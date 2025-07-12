import sys
import torch
from tqdm import tqdm
import numpy as np


def evaluate_model(model, test_loader, device):
    """
    评估模型，并返回真实标签和预测概率，用于计算 ROC AUC。

    参数:
        model (nn.Module): 训练好的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 设备（CPU 或 GPU）。

    返回:
        y_true (np.ndarray): 所有测试样本的真实标签。
        y_scores (np.ndarray): 所有测试样本预测为正类的概率。
        accuracy (float): 测试集准确率。
    """
    model.eval()  # 设置模型为评估模式
    total_correct = 0
    total_testset = 0
    y_true = []
    y_scores = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for data in test_bar:
            points, target = data
            points, target = points.to(device), target.to(device)

            pred, _, _ = model(points)
            pred_choice = pred.data.max(1)[1]  # 预测类别
            pred_prob = torch.softmax(pred, dim=1)[:, 1]  # 预测为正类的概率

            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]

            # 收集真实标签和预测概率
            y_true.append(target.cpu().numpy())
            y_scores.append(pred_prob.cpu().numpy())

    # 合并所有 batch 的结果
    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)

    # 计算准确率
    accuracy = total_correct / float(total_testset)
    print(f"Final accuracy: {accuracy:.4f}")

    return y_true, y_scores, accuracy
