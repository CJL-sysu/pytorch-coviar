"""Training options."""

import argparse

parser = argparse.ArgumentParser(description="CoViAR")

# 数据部分
parser.add_argument(
    "--data-name", type=str, choices=["ucf101", "hmdb51"], help="数据集名称。"
)  # 设置数据集名称的选项，可以是 'ucf101' 或 'hmdb51'。
parser.add_argument(
    "--data-root", type=str, help="数据目录的根路径。"
)  # 设置数据目录的根路径。
parser.add_argument(
    "--train-list", type=str, help="训练样本列表。"
)  # 设置训练样本列表的路径。
parser.add_argument(
    "--test-list", type=str, help="测试样本列表。"
)  # 设置测试样本列表的路径。

# 模型部分
parser.add_argument(
    "--representation",
    type=str,
    choices=["iframe", "mv", "residual"],
    help="数据表示方式。",
)  # 设置数据表示方式，可以是 'iframe'、'mv' 或 'residual'。
parser.add_argument(
    "--arch", type=str, default="resnet152", help="基础架构。"
)  # 设置基础模型架构，默认为 'resnet152'。
parser.add_argument(
    "--num_segments", type=int, default=3, help="TSN（时间分段网络）的段数。"
)  # 设置TSN段数，默认为3。
parser.add_argument(
    "--no-accumulation", action="store_true", help="禁用运动向量和残差的累积。"
)  # 禁用运动向量和残差的累积。

# 训练部分
parser.add_argument(
    "--epochs", default=500, type=int, help="训练周期数。"
)  # 设置训练周期数，默认为500。
parser.add_argument(
    "--batch-size", default=40, type=int, help="批量大小。"
)  # 设置批量大小，默认为40。
parser.add_argument(
    "--lr", default=0.001, type=float, help="基础学习率。"
)  # 设置基础学习率，默认为0.001。
parser.add_argument(
    "--lr-steps",
    default=[200, 300, 400],
    type=float,
    nargs="+",
    help="学习率衰减的周期。",
)  # 设置学习率衰减的周期，默认为[200, 300, 400]。
parser.add_argument(
    "--lr-decay", default=0.1, type=float, help="学习率衰减因子。"
)  # 设置学习率衰减因子，默认为0.1。
parser.add_argument(
    "--weight-decay", "--wd", default=1e-4, type=float, help="权重衰减。"
)  # 设置权重衰减，默认为1e-4。

# 日志部分
parser.add_argument(
    "--eval-freq", default=5, type=int, help="评估频率（以周期为单位）。"
)  # 设置评估频率，默认为每5个周期。
parser.add_argument(
    "--workers", default=8, type=int, help="数据加载器的工作进程数。"
)  # 设置数据加载器的工作进程数，默认为8。
parser.add_argument(
    "--model-prefix", type=str, default="model", help="模型名称的前缀。"
)  # 设置模型名称的前缀，默认为"model"。
parser.add_argument(
    "--gpus", nargs="+", type=int, default=None, help="GPU编号。"
)  # 设置GPU编号，默认为None。
parser.add_argument(
    "--weights",  type=str, default=None, help="权重路径"
)