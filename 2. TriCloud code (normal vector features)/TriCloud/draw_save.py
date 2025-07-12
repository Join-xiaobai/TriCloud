import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


def plot_and_save_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, y_true, y_pred,
                                  save_dir='1'):
    """
    可视化训练和验证的损失及准确率曲线，并计算统计指标。
    参数:
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。
        train_accuracies (list): 训练准确率列表。
        val_accuracies (list): 验证准确率列表。
        y_true (np.ndarray): 真实标签。
        y_pred (np.ndarray): 预测标签。
        save_dir (str): 保存结果的目录。
    """
    dir = 'save'
    save_dir = os.path.join(dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # 创建画布
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.show()

    # 计算统计指标
    y_pred = (y_pred > 0.5).astype(int)
    precision = precision_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # 保存统计指标到文件
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    # 打印统计指标
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Metrics saved to {os.path.join(save_dir, 'metrics.txt')}")


def calculate_and_save_roc_auc(y_true, y_scores, accuracy, save_dir='1'):
    """
    计算 ROC AUC 并保存结果到文件。

    参数:
        y_true (np.ndarray): 真实标签。
        y_scores (np.ndarray): 预测为正类的概率。
        accuracy (float): 测试集准确率。
        save_dir (str): 保存结果的目录。
    """
    # 确保保存目录存在
    dir = 'save'
    save_dir = os.path.join(dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 计算 ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)

    # 保存结果到文件
    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Results saved to {os.path.join(save_dir, 'results.txt')}")


def plot_roc_curve(y_true, y_scores, save_dir='1'):
    """
    绘制ROC曲线并保存图像。

    参数:
        y_true (np.ndarray): 真实标签。
        y_scores (np.ndarray): 预测为正类的概率。
        save_dir (str): 保存图像的目录。
    """
    # 确保保存目录存在
    dir = 'save'
    save_dir = os.path.join(dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 计算ROC曲线的点
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 保存图像
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.show()





