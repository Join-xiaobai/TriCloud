# config.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="PointNet Point-wise Classification Training")

    parser.add_argument('--data_dir', type=str, default='./data/drug', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--in_channels', type=int, default=6, help='Input channels (xyz + normal)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--mat_diff_loss_scale', type=float, default=0.001,
                        help='Regularization scale for feature transform matrix')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Model save path')
    parser.add_argument('--save_freq', type=int, default=1, help='Save model every N epochs')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use GPU if available')

    return parser.parse_args()