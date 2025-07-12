import os
import sys
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    precision_recall_curve, roc_curve, auc

from pointnet_D_T_S_repeat.model import PointNetCls
from pointnet_D_T_S_repeat.dataload import CustomDataset


def evaluate_model(model, test_loader, device, save_path='save/metrix_1.txt'):
    """
    评估模型，并返回多个评估指标。

    参数:
        model (nn.Module): 训练好的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 设备（CPU 或 GPU）。

    返回:
        dict: 包含以下指标的字典:
            - accuracy (float): 测试集准确率
            - precision (float): 精确率
            - recall (float): 召回率
            - f1 (float): F1分数
            - auc (float): ROC AUC
            - aupr (float): PR AUC
            - y_true (np.ndarray): 真实标签
            - y_scores (np.ndarray): 预测概率
    """
    model.eval()  # 设置模型为评估模式
    total_correct = 0
    total_testset = 0
    y_true = []
    y_scores = []
    y_pred = []

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

            # 收集真实标签、预测类别和预测概率
            y_true.append(target.cpu().numpy())
            y_pred.append(pred_choice.cpu().numpy())
            y_scores.append(pred_prob.cpu().numpy())

    # 合并所有 batch 的结果
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_scores = np.concatenate(y_scores)

    # 计算各项指标
    accuracy = total_correct / float(total_testset)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"PR AUC: {aupr:.4f}")
    # 保存指标到txt文件
    with open(save_path, 'w') as f:
        f.write("Model Evaluation Metrics:\n")
        f.write("========================\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"ROC AUC: {auc:.4f}\n")
        f.write(f"PR AUC (AUPR): {aupr:.4f}\n")
        f.write("\nAdditional Info:\n")
        f.write(f"Total samples evaluated: {total_testset}\n")

    print(f"\nEvaluation metrics saved to {save_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'aupr': aupr,
        'y_true': y_true,
        'y_scores': y_scores
    }


def plot_roc_aupr_curves(results_dict, save_dir='save/curves'):
    """
    绘制多个模型的ROC和AUPR曲线

    参数:
        results_dict (dict): 包含模型名称和评估结果的字典，格式为 {'model_name': results}
        save_dir (str): 图片保存目录
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 创建图形
    plt.figure(figsize=(15, 6))

    # 1. ROC曲线图
    plt.subplot(1, 2, 1)
    for model_name, results in results_dict.items():
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_scores'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 2. PR曲线图
    plt.subplot(1, 2, 2)
    for model_name, results in results_dict.items():
        precision, recall, _ = precision_recall_curve(results['y_true'], results['y_scores'])
        aupr = average_precision_score(results['y_true'], results['y_scores'])
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AUPR = {aupr:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")

    # 保存图片
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'roc_aupr_curves.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC and AUPR curves saved to {plot_path}")

if __name__ == "__main__":

    pos_file = '../data/data_0.txt'
    neg_file = '../data/data_1.txt'
    train_dataset = CustomDataset(pos_file, neg_file, split='train')
    val_dataset = CustomDataset(pos_file, neg_file, split='val')
    test_dataset = CustomDataset(pos_file, neg_file, split='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    feature_transform = False
    classifier1 = PointNetCls(k=2, feature_transform=feature_transform)
    classifier2 = PointNetCls(k=2, feature_transform=feature_transform)
    classifier3 = PointNetCls(k=2, feature_transform=feature_transform)
    classifier4 = PointNetCls(k=2, feature_transform=feature_transform)
    classifier5 = PointNetCls(k=2, feature_transform=feature_transform)
    classifier6 = PointNetCls(k=2, feature_transform=feature_transform)

    print("Loading model 1...")
    print(torch.load('../checkpoints/best_model_1.pth'))
    print(classifier1)
    classifier1.to(device)
    classifier1.load_state_dict(torch.load('../checkpoints/best_model_1.pth'))
    classifier2.to(device)
    classifier2.load_state_dict(torch.load('../checkpoints/best_model_2.pth'))
    classifier3.to(device)
    classifier3.load_state_dict(torch.load('../checkpoints/best_model_3.pth'))
    classifier4.to(device)
    classifier4.load_state_dict(torch.load('../checkpoints/best_model_4.pth'))
    classifier5.to(device)
    classifier5.load_state_dict(torch.load('../checkpoints/best_model_5.pth'))
    classifier6.to(device)
    classifier6.load_state_dict(torch.load('../checkpoints/best_model_6.pth'))


    # 评估模型
    results1 = evaluate_model(classifier1, test_loader, device, save_path='save/metrix_1.txt')
    results2 = evaluate_model(classifier2, test_loader, device, save_path='save/metrix_2.txt')
    results3 = evaluate_model(classifier3, test_loader, device, save_path='save/metrix_3.txt')
    results4 = evaluate_model(classifier4, test_loader, device, save_path='save/metrix_4.txt')
    results5 = evaluate_model(classifier5, test_loader, device, save_path='save/metrix_5.txt')
    results6 = evaluate_model(classifier6, test_loader, device, save_path='save/metrix_6.txt')

    all_results = {
        'Model1': results1,
        'Model2': results2,
        'Model3': results3,
        'Model4': results4,
        'Model5': results5,
        'Model6': results6
    }

    # 绘制ROC和AUPR曲线
    plot_roc_aupr_curves(all_results)