# ContactVision: 从视频中学习脚步接触状态

[Daeyong Kim](https://github.com/DaeeYong), [Gyuseok Yi](https://github.com/yigyu), [Ri Yu](https://yul85.github.io/)

![Teaser Image](figures/teaser.jpg)

韩国亚洲大学 (Ajou University, South Korea)

---

## 项目简介

ContactVision 是一个基于深度学习的脚步-地面接触状态检测框架。它能直接从视频中检测脚跟 (heel) 和脚尖 (toe) 的接触状态，无需压力垫或测力台等额外硬件。

核心模型 `FootContactTransformer` 是一个 Transformer 编码器，输入 OpenPose 提取的下半身关节坐标（相对于盆骨的相对坐标），输出每帧 4 个二值标签：`['left_toe', 'right_toe', 'left_heel', 'right_heel']`。

**应用场景：**
- **步态动画重建**：将预测的接触标签用于强化学习框架，实现真实的脚步运动
- **步态分析**：估算临床步态参数（如双支撑时间、单支撑时间）

---

## 环境要求

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- 需要预先安装 [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 来处理视频，生成 2D 骨架关键点 JSON 文件

---

## 安装

本项目使用 **uv** 作为包管理器。如果你还没有安装 uv，请先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. 克隆项目

```bash
git clone https://github.com/DaeeYong/ContactVision.git
cd ContactVision
```

### 2. 使用 uv 创建虚拟环境并安装依赖

```bash
uv sync
```

这将会：
- 根据 `pyproject.toml` 创建虚拟环境
- 自动安装所有依赖（torch、numpy、opencv-python、colorama）

### 3. （可选）以可编辑模式安装

如果你想开发或修改代码，可以以可编辑模式安装：

```bash
uv pip install -e .
```

安装后，你可以直接使用以下命令（在任何目录下运行）：

```bash
contactvision-inference --help
contactvision-preprocess --help
contactvision-opjson2npy --help
contactvision-vis-labels --help
contactvision-vis-op --help
```

### 传统 pip 安装

如果你不使用 uv，也可以用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 项目结构

```
ContactVision/
├── pyproject.toml          # 项目配置（uv/pip 安装入口）
├── README.md               # 本文件
├── src/
│   └── contact_vision/     # Python 包
│       ├── __init__.py
│       ├── model.py        # FootContactTransformer 模型定义
│       ├── inference.py    # 推理脚本
│       ├── preprocess.py   # 数据预处理（OpenPose JSON -> pelvis 相对坐标）
│       ├── opjson2npy.py   # OpenPose JSON -> 原始 .npy
│       ├── vis_labels.py   # 在视频上可视化预测标签
│       └── vis_op.py       # 在视频上可视化 OpenPose 关键点
├── checkpoints/
│   └── best_model.pth      # 预训练模型权重
├── data/
│   └── sample/             # 示例数据
│       ├── openpose/       # OpenPose JSON 文件
│       ├── video/          # 原始视频
│       └── 2d_joint_npy/   # 预处理的 .npy 文件
│           ├── raw/        # 原始关键点 (T, 25, 3)
│           └── processed/  # 预处理后 (T, 13, 3)
└── figures/
    └── teaser.jpg
```

---

## 使用流程

完整的使用流程如下：

```
视频 → OpenPose → JSON → opjson2npy → 原始 .npy → preprocess → 处理后的 .npy → inference → 标签 .npy → vis_labels → 可视化视频
```

### 第 1 步：使用 OpenPose 提取关键点

对输入视频运行 OpenPose，生成每帧的 JSON 关键点文件：

```bash
./build/examples/openpose/openpose.bin --video input.mp4 --write_json ./openpose_output/ --display 0
```

输出的 JSON 目录应包含每帧一个文件，如 `frame_000000000000_keypoints.json`。

### 第 2 步：将 OpenPose JSON 转换为 .npy

```bash
contactvision-opjson2npy --input_dir ./data/sample/openpose/4/ --output_path ./raw_keypoints.npy
```

或使用 Python 直接运行：

```bash
python -m contact_vision.opjson2npy --input_dir ./data/sample/openpose/4/ --output_path ./raw_keypoints.npy
```

### 第 3 步：预处理（提取下半身关节 + 盆骨相对坐标）

```bash
contactvision-preprocess --input_dir ./data/sample/openpose/4/ --output_path ./processed_joints.npy
```

处理后的 `.npy` 形状为 `(T, 13, 3)`，包含 13 个下半身关节的 (x, y) 相对坐标（以盆骨为原点）。

### 第 4 步：运行推理

```bash
contactvision-inference --input_path ./processed_joints.npy --output_path ./pred_labels.npy
```

也可指定模型路径：

```bash
contactvision-inference --input_path ./processed_joints.npy --output_path ./pred_labels.npy --model_path ./checkpoints/best_model.pth
```

输出 `.npy` 形状为 `(T, 4)`，每列对应 `['left_toe', 'right_toe', 'left_heel', 'right_heel']`，值为 0 或 1。

### 第 5 步：可视化预测结果

可视化预测标签：

```bash
contactvision-vis-labels --video ./data/sample/video/4.mp4 --pose ./raw_keypoints.npy --labels ./pred_labels.npy --out_path ./output_annotated.mp4
```

可视化 OpenPose 关键点：

```bash
contactvision-vis-op --input_npy ./raw_keypoints.npy --input_video ./data/sample/video/4.mp4 --output_path ./output_op.mp4 --flag 1
```

> `--flag 1` 表示保存视频文件，`--flag 0` 只显示实时预览。

---

## 使用示例数据快速体验

项目提供了 3 组示例数据（`4`、`b_1`、`c_1`），每组包含视频、OpenPose JSON 和预处理好的 .npy 文件。你可以直接使用预处理好的数据进行推理：

```bash
# 进入项目根目录
cd ContactVision

# 直接推理（使用已预处理好的数据）
contactvision-inference \
    --input_path ./data/sample/2d_joint_npy/processed/4.npy \
    --output_path ./pred_4.npy

# 可视化预测结果
contactvision-vis-labels \
    --video ./data/sample/video/4.mp4 \
    --pose ./data/sample/2d_joint_npy/raw/4.npy \
    --labels ./pred_4.npy \
    --out_path ./output_4.mp4
```

---

## 模型输出说明

模型对每帧输出 4 个二值标签（0 或 1），顺序为：

| 索引 | 标签 | 含义 |
|------|------|------|
| 0 | `left_toe` | 左脚脚尖着地 |
| 1 | `right_toe` | 右脚脚尖着地 |
| 2 | `left_heel` | 左脚跟着地 |
| 3 | `right_heel` | 右脚跟着地 |

---

## 常见问题

**Q: 为什么推理时要进入项目根目录运行？**

A: 因为默认的模型路径 `./checkpoints/best_model.pth` 是相对于当前工作目录的。建议在项目根目录下运行所有命令，或通过 `--model_path` 指定绝对路径。

**Q: 可以使用 CPU 运行吗？**

A: 可以。代码会自动检测 CUDA，如果没有 GPU 会回退到 CPU。但推理速度会较慢。

**Q: 如何处理自己的视频？**

A: 首先用 OpenPose 提取关键点，然后按上述"使用流程"中的步骤 2-5 依次处理即可。

---

## 引用

如果你在研究中使用了本项目，请引用原始论文：

```bibtex
@article{kim2025contactvision,
  title={ContactVision: Learning Foot Contact from Video for Physically Plausible Gait Animation},
  author={Kim, Daeyong and Yi, Gyuseok and Yu, Ri},
  year={2025}
}
```

---

## 许可证

本项目基于 MIT 许可证开源。
