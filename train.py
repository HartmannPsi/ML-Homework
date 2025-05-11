# FixMatch 实现：MNIST + EMNIST 半监督分类任务
# Features:
# 1. 可选多种模型 (CNN, ResNet18, MobileNetV2, ShuffleNetV2, ViT)
# 2. 动态伪标签阈值
# 3. Early Stopping 稳定性优化
# 4. 对有标签数据集使用 RandAugment 增强并增广
# 5. *DEPRECATE*: 使用 EMA 模型预测伪标签
# 6. 使用 SGD 代替 Adam 优化器
# 7. 使用动态平滑学习率

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os
import time
from lib import *

learning_rate = 0.03
round = 50
lambda_unsupervised = 1.0
max_threshold = 0.95
min_threshold = 0.5
thres_arg = 50
seed = 42
batch_labeled = 64
batch_unlabeled = 128

load_saved_model = False
save_model = True
save_log = False
calc_params = False
labeled_data_augmentation = True

model_name = "ResNet"
model_save_path = f"fixmatch_mnist_model_{model_name}.pth"
log_save_path = f"log_{model_name}.txt"

# 设置种子
set_seed(seed=seed)

# 主函数
def main():
    # 初始化有标签数据集、无标签数据集和验证集
    labeled_set = get_labeled_mnist_augmented() if labeled_data_augmentation else get_labeled_mnist()
    unlabeled_set = get_unlabeled_data()
    test_set = get_test_set()

    labeled_loader = DataLoader(labeled_set, batch_size=batch_labeled, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_unlabeled, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # 初始化模型与优化器
    if model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "ResNet":
        model = ResNet18()
    elif model_name == "MobileNet":
        model = MobileNetV2()
    elif model_name == "ShuffleNet":
        model = ShuffleNetV2()
    elif model_name == "ViT":
        model = ViT()
    else:
        print(f"Undefined Model: {model_name}")
        return 1
    model = model.to(device)

    if calc_params:
        print(f"Model {model_name} Params: {count_parameters(model)}")
        return 0

    # 可选择加载已保存的模型
    if os.path.exists(model_save_path) and load_saved_model:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Load Model from: {model_save_path}")
    ema_model = 0
    # ema_model = deepcopy(model).to(device)
    # ema_model.requires_grad_(False)
    # ema_model.eval()

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(
        model.parameters(),       # 模型参数
        lr=learning_rate,         # 学习率
        momentum=0.9,             # 动量
        weight_decay=5e-4,        # L2正则
        nesterov=True             # Nesterov 动量
    )

    # 初始化损失数组与最优准确率
    train_losses = []
    sup_losses = []
    unsup_losses = []
    total_train_time = 0
    total_eval_time = 0
    save_threshold = 95
    save_step = 0.1
    max_acc = 0
    max_acc_epoch = 0

    # 将训练日志重定向至日志文件
    if save_log:
        sys.stdout = open(log_save_path, "w", encoding="utf-8")

    # 训练预设轮数
    for epoch in range(1, round + 1):
        train_start = time.time()
        total_loss, sup_loss, unsup_loss = train_fixmatch(model, ema_model, labeled_loader,
                                                          unlabeled_loader, optimizer, epoch,
                                                          round, lambda_u=lambda_unsupervised,
                                                          max_threshold=max_threshold,
                                                          min_threshold=min_threshold,
                                                          thres_arg=thres_arg,
                                                          dynamic_thres=not load_saved_model,
                                                          thres_strategy=thres_smooth,
                                                          enable_mixup=False,
                                                          mu=batch_unlabeled / batch_labeled)
        train_end = time.time()
        total_train_time += train_end - train_start

        train_losses.append(total_loss)
        sup_losses.append(sup_loss)
        unsup_losses.append(unsup_loss)

        if epoch > round / 2:
            eval_start = time.time()
            acc = evaluate(model, test_loader)
            eval_end = time.time()
            total_eval_time += eval_end - eval_start
            if acc > max_acc:
                max_acc, max_acc_epoch = acc, epoch

            # 保存最优模型参数
            if acc > save_threshold and save_model:
                save_threshold = min(max(99.9, acc), acc + save_step)
                torch.save(model.state_dict(), model_save_path)
                print(f"Model Saved at: {model_save_path} with Accuracy: {acc:.2f}%")
    
    # 输出用时
    print(f"Total Time (sec): {total_eval_time + total_train_time:.2f}; Train Time: {total_train_time:.2f}; Eval Time: {total_eval_time:.2f};"
          f" Max Accuracy: {max_acc:.2f}% at Epoch {max_acc_epoch:.0f}")
    print(f"Labeled Dataset Size: {len(labeled_set):.0f}; Unlabeled Dataset Size: {len(unlabeled_set):.0f}")
    
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
