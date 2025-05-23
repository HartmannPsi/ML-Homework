import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, EMNIST
from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2, shufflenet_v2_x0_5
import numpy as np
import random
from timm import create_model


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 设置设备（GPU优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义弱增强（Weak Augmentation）和强增强（Strong Augmentation）策略
weak_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor()
])

strong_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=4),
    transforms.RandAugment(),
    transforms.ToTensor()
])

# 定义有标签数据集增强策略
aug_transform = transforms.Compose([
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

# 定义增强类
class AugmentedLabeledMNIST(Dataset):
    def __init__(self, base_dataset, n_aug=4, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.n_aug = n_aug

        for img, label in base_dataset:
            # 原始图像保留
            self.images.append(img)
            self.labels.append(label)
            # 添加增强版本
            for _ in range(n_aug):
                self.images.append(img)
                self.labels.append(label)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.images)

# 加载 MNIST 数据集并对每个数据进行增强
def get_labeled_mnist_augmented(n_aug=4):
    full_mnist = MNIST(root="./dataset", train=True, download=True)
    class_counts = {i: 0 for i in range(10)}
    indices = []
    for i, (_, label) in enumerate(full_mnist):
        if class_counts[label] < 20:
            indices.append(i)
            class_counts[label] += 1
        if sum(class_counts.values()) == 200:
            break
    subset = Subset(full_mnist, indices)

    return AugmentedLabeledMNIST(subset, n_aug=n_aug, transform=aug_transform)

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

# MobileNetV2模型
class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = mobilenet_v2(num_classes=10)
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.base(x)

# ShuffleNetV2模型
class ShuffleNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = shufflenet_v2_x0_5(num_classes=10)
        self.base.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.base(x)

# ViT模型
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            in_chans=1,
            num_classes=10
        )

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear")
        return self.base(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 两种动态阈值策略（线性或平滑）
def thres_linear(max_t, min_t, epoch, k):
    step = (max_t - min_t) / k
    return min(max_t, min_t + (epoch - 1) * step)

def thres_smooth(max_t, min_t, epoch, k):
    thres = min_t + (1 - min_t) * (1 - np.exp(-(epoch - 1) / k))
    return min(max_t, thres)

# 余弦学习率衰减策略
def cosine_fixmatch_lr_schedule(base_lr, epoch, round):
    return base_lr * np.cos((7 * np.pi * epoch) / (16 * round))

# FixMatch 训练函数
def train_fixmatch(model, ema_model, labeled_loader, unlabeled_loader,
                   optimizer, epoch, round, lambda_u=1.0, max_threshold=0.95,
                   min_threshold=0.5, thres_arg=50, dynamic_thres=True,
                   thres_strategy=thres_linear,
                   ema_decay=0.999, mu=1):
    model.train()
    # ema_model.eval()
    total_loss, total_supervised, total_unsupervised = 0, 0, 0

    # 动态阈值
    threshold = thres_strategy(max_threshold, min_threshold, epoch, thres_arg) if dynamic_thres else max_threshold

    for batch_idx, ((x_l, y_l), (x_uw, x_us)) in enumerate(zip(labeled_loader, unlabeled_loader)):
        x_l, y_l = x_l.to(device), y_l.to(device)
        x_uw, x_us = x_uw.to(device), x_us.to(device)

        # 有标签监督损失
        logits_l = model(x_l)
        loss_l = F.cross_entropy(logits_l, y_l)

        # 弱增强预测伪标签
        with torch.no_grad():
            pseudo_labels = torch.softmax(model(x_uw), dim=1)
            max_probs, targets_u = torch.max(pseudo_labels, dim=1)
            mask = max_probs.ge(threshold).float()
            # print(f"Max prob: {max_probs.mean().item():.4f}, used: {(max_probs > threshold).sum().item()}")

        # 强增强计算无标签伪监督损失
        logits_u_s = model(x_us)
        loss_u = (1 / mu) * (F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask).mean()

        # 总损失
        loss = loss_l + lambda_u * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA 模型参数更新
        # with torch.no_grad():
        #     for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        #         ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)
            
        #     for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
        #         ema_buffer.data.copy_(model_buffer.data)
        
        total_loss += loss.item()
        total_supervised += loss_l.item()
        total_unsupervised += loss_u.item()

        # 学习率衰减
        lr = cosine_fixmatch_lr_schedule(base_lr=0.03, epoch=epoch, round=round)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}, Supervised: {total_supervised:.4f}, Unsupervised: {total_unsupervised:.4f}, Used Unlabeled Data: {mask.sum().item():.0f}")
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