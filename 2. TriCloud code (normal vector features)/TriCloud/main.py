import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from pointnet_D_T_S_repeat.model import PointNetCls
from dataload import CustomDataset
from train import train_model
from eval import evaluate_model
from draw_save import plot_and_save_training_curves, calculate_and_save_roc_auc, plot_roc_curve


def main():
    repeat = 1 # 是否复制点，使用多点融合 (值为1表示不融合)
    feature_transform = False
    classifier = PointNetCls(k=2, feature_transform=feature_transform)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    pos_file = './data/data_0.txt'
    neg_file = './data/data_1.txt'
    train_dataset = CustomDataset(pos_file, neg_file, split='train')
    val_dataset = CustomDataset(pos_file, neg_file, split='val')
    test_dataset = CustomDataset(pos_file, neg_file, split='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train the model
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        repeat= repeat,
        num_epochs=200,
        feature_transform=feature_transform,
        scheduler=scheduler,  # 传递 scheduler
        patience=30
    )

    # evaluate the model
    y_true, y_scores, accuracy = evaluate_model(trained_model, test_loader, device)
    # plot and save the training curves
    plot_and_save_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, y_true, y_scores, save_dir = str(repeat))
    calculate_and_save_roc_auc(y_true, y_scores, accuracy, save_dir=str(repeat))
    plot_roc_curve(y_true, y_scores, save_dir = str(repeat))


if __name__ == '__main__':
    main()
