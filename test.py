import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from lib import *

model_name = "ResNet"
model_save_path = f"fixmatch_mnist_model_{model_name}.pth"
log_save_path = f"log_{model_name}.txt"

def show_wrong(wrong, n=9):
    sample = random.sample(wrong, min(n, len(wrong)))
    plt.figure(figsize=(10, 10))
    for idx, (img, label, pred) in enumerate(sample):
        plt.subplot(3, 3, idx + 1)
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {label}, Pred: {pred}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("wrong_preds.png")
    plt.show()

# 主函数
def main():
    # 初始化测试集
    test_set = get_test_set()
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # 初始化模型
    if not os.path.exists(model_save_path):
        print(f"Cannot Load Model from: {model_save_path}")
        return 1

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
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Load Model from: {model_save_path}")

    # 将训练日志重定向至日志文件
    # if save_log:
    #     sys.stdout = open(log_save_path, "w", encoding="utf-8")

    model.eval()
    correct = 0
    total = 0
    wrong = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            for img, label, pr in zip(x.cpu(), y.cpu(), pred.cpu()):
                if label != pr:
                    wrong.append((img, label.item(), pr.item()))
    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%, Total Wrong: {len(wrong)}")
    show_wrong(wrong)

if __name__ == "__main__":
    main()
