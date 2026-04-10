# CV Project README

本仓库包含两个阶段的超分辨率实验：

- Part 1: 单帧图像超分（SRCNN）
- Part 2: 视频超分（BasicVSR 风格简化版）

数据组织采用 REDS/DAVIS 风格目录，代码基于 PyTorch。

## 1. 项目结构

```text
CV/
  requirements.txt
  data/
    train/
      train_sharp/
      train_sharp_bicubic/
    val/
      val_sharp/
      val_sharp_bicubic/
  part1/
    config.yaml
    main.py
    src/
  part2/
    config.yaml
    main.py
    src/
```

## 2. 环境安装

建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- torch
- torchvision
- pyyaml
- pillow
- tqdm

## 3. 数据准备

请确保 LR/HR 帧文件名一一对应。

### 3.1 Part 1 数据格式

`part1/src/dataset.py` 会自动在 `lr_root` 后拼接 `X{scale}`，例如 `scale=4` 时：

- HR: `data/train/train_sharp/000/00000000.png`
- LR: `data/train/train_sharp_bicubic/X4/000/00000000.png`

因此 `part1/config.yaml` 中的 `train_lr / val_lr / test_lr` 应指向 `.../train_sharp_bicubic` 这一级，不要手动写到 `X4`。

### 3.2 Part 2 数据格式

`part2/src/dataset.py` 不会自动拼接 `X{scale}`，所以配置里应直接给到 `.../X4`：

- `train_lr: /.../train_sharp_bicubic/X4`
- `val_lr: /.../val_sharp_bicubic/X4`
- `test_lr: /.../val_sharp_bicubic/X4`

目录示例：

```text
data/train/train_sharp_bicubic/X4/000/00000000.png
data/train/train_sharp/000/00000000.png
```

## 4. Part 1: SRCNN 单帧超分

入口：`part1/main.py`

支持模式：

- `train`: 训练 SRCNN
- `test`: 在测试集评估并保存可视化
- `infer`: 对单张图做推理
- `temporal`: 时域 baseline（邻域帧加权 + 可选锐化）

### 4.1 训练

```bash
cd part1
python main.py --mode train --cfg config.yaml
```

输出：

- checkpoint: `part1/checkpoints/srcnn_x{scale}_epoch{n}.pth`
- 日志目录: `part1/outputs/logs`

### 4.2 测试

```bash
cd part1
python main.py --mode test --cfg config.yaml --ckpt checkpoints/srcnn_x4_epoch20.pth
```

如不传 `--ckpt`，会自动尝试使用 `checkpoints` 中按文件名排序后的最后一个 `.pth`。

测试输出：

- 控制台打印 Bicubic 与 SRCNN 的平均 PSNR
- 可视化图像保存到 `part1/outputs/vis`

### 4.3 单图推理

```bash
cd part1
python main.py --mode infer --cfg config.yaml \
  --ckpt checkpoints/srcnn_x4_epoch20.pth \
  --input /path/to/image.png \
  --output outputs/infer_sr.png
```

未指定 `--output` 时，默认保存到 `paths.out_dir`。

### 4.4 时域 baseline（temporal）

```bash
cd part1
python main.py --mode temporal --cfg config.yaml
```

逻辑：读取邻域 LR 帧 -> 双三次上采样 -> 加权融合 -> 可选 UnsharpMask。

相关参数在 `part1/config.yaml` 的 `temporal` 段：

- `lr_root`
- `out_dir`
- `radius`
- `weights`
- `apply_unsharp`

## 5. Part 2: 视频超分（BasicVSRTiny）

入口：`part2/main.py`

支持模式：

- `train`
- `test`

模型位置：`part2/src/models/basicvsr.py`

当前实现是可训练的 BasicVSR 风格简化版，包含双向传播与上采样重建；`spynet.py` 里是零光流占位模块（`SpyNetStub`）。

### 5.1 训练

```bash
cd part2
python main.py --mode train --cfg config.yaml
```

输出：

- 最新权重：`part2/checkpoints/latest.pth`
- 最优权重：`part2/checkpoints/best.pth`

### 5.2 测试

```bash
cd part2
python main.py --mode test --cfg config.yaml
```

测试阶段会读取 `test.checkpoint`（默认 `checkpoints/best.pth`），并将结果按视频文件夹保存到 `paths.output_dir`。

示例输出结构：

```text
part2/outputs/
  000/
    00000000.png
    00000001.png
```

## 6. 配置说明与注意事项

### 6.1 Part 1

- `scale`: 上采样倍率（当前代码支持 x2/x4 数据组织）
- `patch_size`: 训练裁剪 HR patch 尺寸
- `train.loss`: `l1` 或 `mse`
- `paths.*`: 训练/验证/测试数据路径与输出路径

### 6.2 Part 2

当前 `part2/src/train.py` 实际读取的关键字段：

- `train.batch_size`
- `train.epochs`
- `train.lr`
- `train.weight_decay`（可选）
- `train.log_interval`（可选，默认 10）
- `train.val_interval`（可选，默认 1）
- `seq_len`
- `scale`
- `paths.train_lr / train_hr / val_lr / val_hr / ckpt_dir`
- `model.mid_channels / model.num_blocks`

注意：

- `part2/config.yaml` 里的 `patch_size`、`train.val_every`、`dataset.*` 等字段当前训练脚本并未使用。
- 如需随机裁剪，请在配置中增加 `crop_size`，因为训练代码读取的是 `cfg.get('crop_size')`。

## 7. 常见问题

1. 报错 `No LR/HR pairs found`（Part 1）

请检查：

- `paths.train_lr` 是否指向 `.../train_sharp_bicubic`（不是 `.../X4`）
- `scale` 与实际 LR 子目录是否一致（如 `X4`）
- LR/HR 帧文件名是否完全对应

2. 报错 `Frame mismatch in video ...`（Part 2）

说明某个视频文件夹里 LR 与 HR 帧名不一致，需要对齐命名和数量。

3. GPU 不可用

把配置中的 `device` 从 `cuda` 改为 `cpu`。

## 8. 快速命令清单

```bash
# Part 1 train/test
cd part1
python main.py --mode train --cfg config.yaml
python main.py --mode test --cfg config.yaml --ckpt checkpoints/srcnn_x4_epoch20.pth

# Part 2 train/test
cd ../part2
python main.py --mode train --cfg config.yaml
python main.py --mode test --cfg config.yaml
```
