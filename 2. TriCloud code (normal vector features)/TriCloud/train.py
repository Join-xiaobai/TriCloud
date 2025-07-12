import os
import logging
from itertools import repeat

import torch
import torch.nn.parallel
import torch.utils.data
from model import feature_transform_regularizer
from tqdm import tqdm
import sys


def train_model(model, train_loader, val_loader, criterion, optimizer, repeat = 3, num_epochs=300, feature_transform=False, scheduler=None, patience = 15):
    repeat = repeat  # 是否复制点，使用多点融合 (值为1表示不融合)

    # 确保保存模型的目录存在
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    logging.basicConfig(filename='log/training_'+str(repeat)+'.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_steps = len(train_loader)
    best_val_accuracy = 0.0

    # 定义最佳模型路径
    best_model_path = 'checkpoints/best_model_'+ str(repeat) +'.pth'  # 保存最佳模型的路径
    best_model_state = None  # 用于保存最佳模型的状态字典
    # 早停参数
    patience = patience  # 连续多少个 epoch 没有提升时停止
    no_improvement_count = 0  # 记录没有提升的 epoch 数量
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for batch_idx, (points, target) in enumerate(train_bar):
            # points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            pred, trans, trans_feat = model(points)
            loss = criterion(pred, target)

            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            train_bar.desc = (f"Epoch {epoch + 1}/{num_epochs}, "
                              f"Batch {batch_idx + 1}/{train_steps}, "
                              f"Train Loss: {running_loss / (batch_idx + 1):.4f}")

        train_accuracy = correct_train / total_train
        train_losses.append(running_loss / train_steps)
        train_accuracies.append(train_accuracy)

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, "
                     f"Train Loss: {train_losses[-1]:.4f}, "
                     f"Train Accuracy: {train_accuracy:.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}")
        # 调用 scheduler.step() 更新学习率
        scheduler.step()
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for batch_idx, (points, target) in enumerate(val_bar):
                # points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)

                pred, _, _ = model(points)
                loss = criterion(pred, target)
                val_loss += loss.item()

                _, predicted = torch.max(pred, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

                val_bar.desc = f"Validation Batch {batch_idx + 1}/{len(val_loader)}"

        val_accuracy = correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        logging.info(f"Validation Loss: {val_losses[-1]:.4f}, "
                     f"Validation Accuracy: {val_accuracy:.4f}")

        print(f"Validation Loss: {val_losses[-1]:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")
        # 早停逻辑
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, best_model_path)
            print(f"Epoch {epoch + 1}: 保存最佳模型，验证集准确率: {best_val_accuracy:.4f}")
            # no_improvement_count = 0  # 重置计数器
        # else:
        #     no_improvement_count += 1  # 没有提升，计数器加 1
        # # 如果连续 patience 个 epoch 没有提升，停止训练
        # if no_improvement_count >= patience:
        #     print(f"早停：连续 {patience} 个 epoch 验证集准确率没有提升，停止训练。")
        #     break

    return model, train_losses, val_losses, train_accuracies, val_accuracies


