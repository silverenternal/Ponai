# PointPillars 真实模型集成指南

## 概述

我们已成功从 OpenMMLab 下载了真实的 PointPillars 预训练模型：
- **文件**: `python_tools/models/pointpillars_kitti_3class.pth`
- **大小**: 18.51 MB
- **数据集**: KITTI
- **类别**: Pedestrian, Cyclist, Car (3 类)
- **来源**: MMDetection3D 官方预训练权重

## 下载链接

```
https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
```

## 环境要求

由于 MMDetection3D 与 Python 3.14 不兼容，我们使用 Python 3.11：

```bash
# 创建 Python 3.11 虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 安装 PyTorch CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install onnxruntime onnx fastapi uvicorn scipy numpy==1.26.4
```

## 使用方法

### 方法 1: 使用简化的 PointPillars 实现（推荐用于测试）

```bash
unset LD_LIBRARY_PATH  # 避免系统 libtorch 冲突
source .venv/bin/activate
python python_tools/pointpillars_simple.py
```

**性能**: ~2.7 FPS (CPU, 未优化实现)

### 方法 2: 使用真实权重的 MMDetection3D

需要先安装 MMDetection3D 及其依赖：

```bash
# 安装基础依赖
pip install mmengine mmdet

# 安装 mmcv (需要预编译版本)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# 安装 MMDetection3D
pip install mmdetection3d/ --no-build-isolation --no-deps
```

然后运行推理：

```bash
unset LD_LIBRARY_PATH
source .venv/bin/activate
python python_tools/test_pointpillars_pytorch.py
```

## 模型对比

| 模型 | 大小 | 参数量 | FPS (CPU) | 说明 |
|------|------|--------|-----------|------|
| pointpillars.onnx | 22KB | 5.2K | 6000+ | 简化演示模型 |
| pointpillars_realistic.onnx | 192KB | 48.4K | 3400+ | 真实架构模型 |
| pointpillars_kitti_3class.pth | 18.51MB | ~5.6M | ~100* | 官方预训练模型 |

*实际 FPS 取决于推理引擎和优化

## 已知问题

1. **Python 3.14 兼容性**: MMDetection3D 不支持 Python 3.14
2. **系统 libtorch 冲突**: `/opt/libtorch` 可能干扰虚拟环境，需要 `unset LD_LIBRARY_PATH`
3. **mmcv 版本**: 需要特定版本的 mmcv，建议使用预编译 wheel

## 下一步

1. **获得真实权重**: 已从 OpenMMLab 下载
2. **模型转换**: 由于 MMDetection3D 的复杂性，ONNX 导出需要 MMDeploy
3. **部署选项**:
   - 使用 PyTorch 直接推理（推荐）
   - 使用 MMDeploy 导出 ONNX/TensorRT
   - 使用简化的 ONNX 模型进行框架测试

## 参考资料

- [MMDetection3D GitHub](https://github.com/open-mmlab/mmdetection3d)
- [PointPillars 配置](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars)
- [MMDeploy 部署工具](https://github.com/open-mmlab/mmdeploy)
