# Pylibs
Self-Learning records of useful libs during my CV research 

Tensorflow 已💀，Pytorch才是正统，其他框架之后再学

## 2025.10

目前主要接触比较多的是图像分类任务，经典的Pipeline为先读取图像转为PIL-Image对象/NumPy数组，再转为Tensor，打包为Dataset（结合不同的augmentation），再用DataLoader包一层（结合不同的sampler），接下来作为model的input，计算loss，反传grad，最后评估，必要时保存模型参数。

所以主要的库应该按如下分类：

Pylibs/
├─ data_io/                          # 数据读取 & 预处理
│  ├─ pillow/                        # Pillow (PIL) 读写图像、基础变换
│  ├─ opencv-python (cv2)/           # OpenCV 读取多格式图像、颜色空间、滤波
│  ├─ imageio/                       # 各种格式的图像读写
│  ├─ torchvision.datasets/          # 内置数据集 + ImageFolder 目录分类读取
│  └─ datasets (huggingface)/        # HuggingFace Datasets，统一加载/切分/流式

├─ augmentation/                     # 数据增强 / 预处理算子
│  ├─ torchvision.transforms/        # PyTorch官方基础增强(裁剪, 翻转, Normalize)
│  ├─ albumentations/                # 强大的图像增强库(仿射, 颜色扰动, 模糊, Cutout)
│  ├─ imgaug/                        # 经典增强库，可自定义pipeline
│  ├─ kornia/                        # 基于PyTorch张量的可微增强(几何, 颜色等)

├─ models_backbones/                 # 模型结构 / 预训练骨干
│  ├─ torchvision.models/            # ResNet, DenseNet, EfficientNet, ViT 等官方实现
│  ├─ timm/                          # Ross Wightman的timm：超多SOTA/变体骨干+预训练权重
│  ├─ lightning-bolts / pl_bolts/    # pytorch-lightning社区模型和自监督骨干
│  └─ transformers/                  # HuggingFace Transformers里的ViT, DeiT, CLIP视觉编码器

├─ training_loops/                   # 训练循环/调度/加速
│  ├─ torch/                         # PyTorch (nn.Module, autograd, optim, DataLoader)
│  ├─ pytorch-lightning/             # 高层封装训练/验证/日志/多GPU
│  ├─ accelerate (HF accelerate)/    # 简化分布式/混合精度训练
│  ├─ deepspeed/                     # 大模型/显存优化训练加速              

├─ loss_metrics/                     # 损失函数 & 指标
│  ├─ torchmetrics/                  # Accuracy, Precision, Recall, F1, AUROC, etc.
│  ├─ sklearn.metrics/               # classification_report, confusion_matrix, ROC, etc.
│  ├─ monai.metrics/                 # 医学特定指标(ROC-AUC, Sensitivity/Specificity等)

├─ logging_monitoring/               # 训练过程记录 / 可视化
│  ├─ tensorboard / torch.utils.tensorboard/
│  ├─ wandb (Weights & Biases)       # 实验可视化/对比
│  ├─ mlflow                         # 训练过程追踪、模型版本
│  └─ rich / tqdm                    # 进度条、控制台可视化

├─ evaluation_analysis/              # 结果分析 / 可解释性
│  ├─ captum/                        # PyTorch可解释性(Grad-CAM, IG, etc.)
│  ├─ torchcam/                      # Grad-CAM/Score-CAM等可视化
│  ├─ shap / lime                    # 特征重要性解释
│  ├─ sklearn.metrics.*              # 混淆矩阵、PR曲线、ROC曲线
│  ├─ matplotlib / seaborn           # 混淆矩阵热图、曲线绘图
│  └─ scikit-image (skimage)         # 图像质量测度/后处理辅助

当然，还有支持程序运行的常见库，解析命令行参数，配置分布式训练，记录必要的artifacts等基础设施。

另外一些比较好用的库，termcolor（可以终端输出特定颜色的log，毕竟全黑太单调了），monai（强大的Medical Imaging的库），imbalanced-learn（类不平衡数据集学习）...

真实世界中的数据集往往是含有噪声且可能类不平衡的，像CIFAR10这种精心造出来的玩具数据集不可求，所以一个算法的性能应该在真实数据集上表现良好，而不是仅仅在CIFAR10/100，Tiny-ImageNet上SOTA就满足了（推广就留给后续研究吧🤣）。

Pursue real-world feasible and data-efficient learning.
