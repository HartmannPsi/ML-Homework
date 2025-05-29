# Semi-supervised Model

## 使用方式
首先使用 conda 构建运行环境：
```shell
conda env create -f environment.yml
```
激活环境后运行 `train.py` 即可开始训练模型，默认超参数见文件，可以随意修改：
```shell
python3 train.py
```
训练结束后会保存损失函数随训练轮数变化的曲线图，同时保存模型参数（如果设置了 `save_model`）。运行 `test.py` 可以从指定的路径加载保存的模型参数并在测试集上验证正确率，运行结束后会打印预测的正确率并且展示一部分预测错误的样本。

*注：本地训练和推理时使用的显卡为RTX 5070 Ti，conda环境中库的版本都能够适配该显卡，但是在其他显卡上可能会有兼容性问题。*