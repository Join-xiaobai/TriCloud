import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import logging
from datetime import datetime

# 从 sklearn 导入评估指标
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# 导入模块（请根据你的项目结构调整）
from model import PointNetCls
from dataset import BinaryLabelDataLoader
from config import get_args
from tqdm import tqdm


def compute_metrics(outputs, labels):
    """
    使用模型输出的 logits 计算分类指标：ACC、Precision、Recall、F1、AUC、AUPR。

    参数:
    - outputs: [B, C]，模型输出的 logits（未 softmax），其中 C 是类别数（二分类为2）
    - labels: [B]，真实标签（0 或 additional_dataSet）

    返回:
    - metrics_dict: 包含各项指标的字典
    """
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 只取正类概率用于 AUC/AUPR
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    acc = (preds == labels).mean()
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)
    except:
        auc = float('nan')
        aupr = float('nan')

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'aupr': aupr
    }


def setup_logger(save_path, groupNum=4):
    """
    配置日志记录器，同时输出到控制台和文件

    参数:
    - save_path: 日志文件保存路径
    """
    log_file = os.path.join(save_path, 'training-6t' + str(groupNum) + '.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def main():
    args = get_args()

    groupNum = 1

    # 设置设备（CPU/GPU）
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now()}] 使用设备: {device}")

    # 创建保存路径
    os.makedirs(args.save_path, exist_ok=True)
    model_save_path = os.path.join(args.save_path,f'model+6t' + str(groupNum))
    os.makedirs(model_save_path, exist_ok=True)

    # 配置日志系统
    setup_logger(args.save_path, groupNum)

    # 打印训练参数信息
    logging.info("开始训练，参数如下：")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")
        print(f"  {k}: {v}")

    # 初始化数据集和数据加载器
    file_0 = '../3.Id_mapping/neg_final.csv'
    file_1 = '../3.Id_mapping/pos_random_final.csv'

    train_dataset = BinaryLabelDataLoader(file_0=file_0, file_1=file_1,
                                          normal_channel=True, split='train', test_size=0.2, seed=42, groupNum=groupNum)
    val_dataset = BinaryLabelDataLoader(file_0=file_0, file_1=file_1,
                                        normal_channel=True, split='test', test_size=0.2, seed=42, groupNum=groupNum)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    input_dim = int(3 + 64/groupNum)
    model = PointNetCls(input_dim=input_dim, feature_transform=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = 0.0  # 用 F1 作为最佳模型选择标准

    # 开始训练
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        all_train_outputs = []
        all_train_labels = []

        # 训练阶段
        train_iter = tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
                          leave=False)
        for points, labels in train_iter:
            points = points.to(device)  # [B, D] -> [B, D, additional_dataSet]
            labels = labels.squeeze().to(device).long()

            outputs, _, _ = model(points)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # 收集 outputs 和 labels（注意：不是预测值，而是模型原始输出）
            all_train_outputs.append(outputs.detach().cpu())
            all_train_labels.append(labels.cpu())
            # 更新进度条显示当前loss
            train_iter.set_postfix(loss=loss.item())
        # 拼接所有 batch 的 outputs 和 labels
        train_outputs = torch.cat(all_train_outputs, dim=0)
        train_labels = torch.cat(all_train_labels, dim=0)

        # 计算训练指标
        train_metrics = compute_metrics(train_outputs, train_labels)
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        all_val_outputs = []
        all_val_labels = []

        val_iter = tqdm(val_loader,
                        desc=f"Epoch {epoch + 1}/{args.epochs} [Val]",
                        leave=False)
        with torch.no_grad():
            for points, labels in val_iter:
                points = points.to(device)
                labels = labels.squeeze().to(device).long()
                # print(labels)

                outputs, _, _ = model(points)
                # print(outputs.shape)
                # print(labels.shape)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                # 收集 outputs 和 labels
                all_val_outputs.append(outputs.detach().cpu())
                all_val_labels.append(labels.cpu())

        # 拼接所有 batch 的 outputs 和 labels
        val_outputs = torch.cat(all_val_outputs, dim=0)
        val_labels = torch.cat(all_val_labels, dim=0)

        # 计算验证指标
        val_metrics = compute_metrics(val_outputs, val_labels)
        avg_val_loss = total_val_loss / len(val_loader)

        # 打印 & 记录日志
        logging.info(f"[Epoch {epoch + 1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}, "
                     f"ACC: {train_metrics['acc']:.4f}, "
                     f"Precision: {train_metrics['precision']:.4f}, "
                     f"Recall: {train_metrics['recall']:.4f}, "
                     f"F1: {train_metrics['f1']:.4f}, "
                     f"AUC: {train_metrics['auc']:.4f}, "
                     f"AUPR: {train_metrics['aupr']:.4f}")

        logging.info(f"[Epoch {epoch + 1}/{args.epochs}] Val Loss: {avg_val_loss:.4f}, "
                     f"ACC: {val_metrics['acc']:.4f}, "
                     f"Precision: {val_metrics['precision']:.4f}, "
                     f"Recall: {val_metrics['recall']:.4f}, "
                     f"F1: {val_metrics['f1']:.4f}, "
                     f"AUC: {val_metrics['auc']:.4f}, "
                     f"AUPR: {val_metrics['aupr']:.4f}")

        print(f"[Epoch {epoch + 1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}, "
              f"ACC: {train_metrics['acc']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, "
              f"AUPR: {train_metrics['aupr']:.4f}")

        print(f"[Epoch {epoch + 1}/{args.epochs}] Val Loss: {avg_val_loss:.4f}, "
              f"ACC: {val_metrics['acc']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"AUPR: {val_metrics['aupr']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1 and (epoch + 1) % args.save_freq == 0:
            best_val_f1 = val_metrics['f1']
            bestModel_save_path = model_save_path + '/best_model.pth'
            torch.save(model.state_dict(), bestModel_save_path)
            logging.info(f"模型已保存至: {model_save_path}")

    logging.info("训练完成.")
    print("✅ 训练完成.")


if __name__ == '__main__':
    main()