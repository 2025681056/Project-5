# Project 1 NumPy Image Encoder 复现说明

本文件是提交用的运行说明。

## 1. 当前结果

`image_encoder.py` 已补全 TODO-01 至 TODO-06。实现只使用 NumPy，调用提供的 `helpers.py` 完成 LayerNorm、精确 GELU、注意力系数和单位向量缩放。模型参数来自 `weights/image_encoder.npz`，参考数组仅由 `check.py` 读取，编码器本身从不读取参考数据。

在附带的十张图片上运行 `check.py --stage all`，全部阶段通过。输出 token 与独立 FP32 参考的最大绝对误差为 `5.439e-07`，批量乱序与重复图片检查的最大绝对误差为 `4.619e-07`。输出形状为 `(10, 512)`，dtype 为 `float32`，每行 L2 范数均为 `1.0`。

## 2. 环境

题目允许 Python 3.9 及以上版本。依赖列在 `requirements.txt`：`numpy>=1.24,<3`，`Pillow>=9.2,<13`，`imageio>=2.31,<3`。实际验证环境为 Python 3.13.14，NumPy 2.5.4，Pillow 13.0.0。Pillow 13.0.0 略高于文件约束上限，resize 与中心裁剪结果已通过 `preprocess` 检查，可按 `requirements.txt` 安装更保守的版本。

## 3. 复现步骤

以下命令均从 `project1-numpy-student` 目录运行。若从其他目录运行，请改用脚本的完整路径。

```bash
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

自定义图片：

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

预期结果：

```text
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS shape=(10, 3, 224, 224) max_abs=0.000e+00
patches      PASS shape=(2, 49, 3072) max_abs=0.000e+00
tokens       PASS shape=(2, 50, 768) max_abs=4.005e-05
block0       PASS shape=(2, 50, 768) max_abs=7.063e-06
last_tokens  PASS shape=(2, 50, 768) max_abs=4.768e-05
output       PASS shape=(10, 512) max_abs=5.439e-07
batch        PASS shape=(3, 512) max_abs=4.619e-07
Requested numerical checks passed. The reference file is never an encoder input.
```

`run.py` 会打印 `Saved (10, 512) to output_tokens.npz`，并在 npz 中保存 `ids` 与 `output_tokens`，顺序与输入一致。
