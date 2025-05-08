# FixMatch 实现：MNIST + EMNIST 半监督分类任务

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import MNIST, EMNIST
from torchvision import transforms
from torchvision.models import resnet18
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# 设置设备（GPU优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义弱增强（Weak Augmentation）和强增强（Strong Augmentation）策略
weak_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor()
])

strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=4),
    transforms.RandAugment(),
    transforms.ToTensor()
])

# 定义基本预处理（用于测试集）
basic_transform = transforms.ToTensor()

# 加载 MNIST 数据集（有标签部分，每类20张，共200张）
def get_labeled_mnist():
    full_mnist = MNIST(root="./dataset", train=True, download=True)
    class_counts = {i: 0 for i in range(10)}
    indices = []
    for i, (_, label) in enumerate(full_mnist):
        if class_counts[label] < 20:
            indices.append(i)
            class_counts[label] += 1
        if sum(class_counts.values()) == 200:
            break
    return Subset(MNIST(root="./dataset", train=True, transform=weak_transform), indices)

# 加载未标注数据（8000 MNIST + 2000 EMNIST-letters）
def get_unlabeled_data():
    unlabeled_mnist = MNIST(root="./dataset", train=True, transform=None)
    unlabeled_emnist = EMNIST(root="./dataset", split="letters", train=True, transform=None)
    
    mnist_subset = Subset(unlabeled_mnist, list(range(8000)))
    emnist_subset = Subset(unlabeled_emnist, list(range(2000)))
    
    # 定义 wrapper 数据集类用于增强	simultaneously weak and strong
    class UnlabeledDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, idx):
            img, _ = self.dataset[idx]
            return weak_transform(img), strong_transform(img)

        def __len__(self):
            return len(self.dataset)

    return UnlabeledDataset(ConcatDataset([mnist_subset, emnist_subset]))

# 加载测试集
def get_test_set():
    return MNIST(root="./dataset", train=False, transform=basic_transform)

# 简单CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ResNet18模型
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = resnet18(weights=None)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base.maxpool = nn.Identity()
        self.base.fc = nn.Linear(self.base.fc.in_features, 10)

    def forward(self, x):
        return self.base(x)

# FixMatch 训练函数
def train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, epoch, lambda_u=1.0, threshold=0.95):
    model.train()
    total_loss, total_supervised, total_unsupervised = 0, 0, 0

    for batch_idx, ((x_l, y_l), (x_uw, x_us)) in enumerate(zip(labeled_loader, unlabeled_loader)):
        x_l, y_l = x_l.to(device), y_l.to(device)
        x_uw, x_us = x_uw.to(device), x_us.to(device)

        # 有标签监督损失
        # 数据增强： one-hot + mixup
        logits_l = model(x_l)
        loss_l = F.cross_entropy(logits_l, y_l)

        # 弱增强预测伪标签
        with torch.no_grad():
            pseudo_labels = torch.softmax(model(x_uw), dim=1)
            max_probs, targets_u = torch.max(pseudo_labels, dim=1)
            mask = max_probs.ge(threshold).float()

        # 无标签伪监督损失
        logits_u_s = model(x_us)
        loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask).mean()

        # 总损失
        loss = loss_l + lambda_u * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_supervised += loss_l.item()
        total_unsupervised += loss_u.item()

    print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}, Supervised: {total_supervised:.4f}, Unsupervised: {total_unsupervised:.4f}, Used Unlabeled Data: {mask.sum().item()}")
    return total_loss, total_supervised, total_unsupervised

# 测试函数
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

learning_rate = 0.001
round = 100
lambda_unsupervised = 1.0
threshold = 0.8

load_saved_model = False
save_model = False
save_log = False

model_name = "ResNet"
model_save_path = f"fixmatch_mnist_model_{model_name}.pth"
log_save_path = "log.txt"

# 主函数
def main():
    # 初始化有标签数据集、无标签数据集和验证集
    labeled_set = get_labeled_mnist()
    unlabeled_set = get_unlabeled_data()
    test_set = get_test_set()

    labeled_loader = DataLoader(labeled_set, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # 初始化模型与优化器，可选用CNN或ResNet
    if model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "ResNet":
        model = ResNet18()
    else:
        print(f"Undefined Model: {model_name}")
        return 1
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 可选择加载已保存的模型
    if os.path.exists(model_save_path) and load_saved_model:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Load Model from: {model_save_path}")


    # 初始化损失数组
    train_losses = []
    sup_losses = []
    unsup_losses = []

    # 将训练日志重定向至日志文件
    if save_log:
        sys.stdout = open(log_save_path, "w", encoding="utf-8")

    # 训练预设轮数
    for epoch in range(1, round + 1):
        total_loss, sup_loss, unsup_loss = train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, epoch, lambda_u=lambda_unsupervised, threshold=threshold)
        evaluate(model, test_loader)
        train_losses.append(total_loss)
        sup_losses.append(sup_loss)
        unsup_losses.append(unsup_loss)

    # 保存模型参数到硬盘
    if save_model:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model Saved at: {model_save_path}")
    
    # 绘制训练损失曲线
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Total Loss")
    plt.plot(epochs, sup_losses, label="Supervised Loss")
    plt.plot(epochs, unsup_losses, label="Unsupervised Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FixMatch Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    main()
