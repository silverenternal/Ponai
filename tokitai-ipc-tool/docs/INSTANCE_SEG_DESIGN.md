# 3D 点云实例分割模型集成方案

**项目**：Lidar AI Studio - tokitai-ipc-tool  
**版本**：v1.0  
**日期**：2026 年 3 月 25 日  
**作者**：AI Assistant

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [现有架构分析](#2-现有架构分析)
3. [模型选型与技术调研](#3-模型选型与技术调研)
4. [详细设计方案](#4-详细设计方案)
5. [实施计划](#5-实施计划)
6. [风险评估与应对](#6-风险评估与应对)
7. [附录](#7-附录)

---

## 1. 项目背景与目标

### 1.1 项目背景

Lidar AI Studio 是一个基于 tokitai 框架构建的 3D 点云车机应用，当前已实现：
- Rust AI 调度层（Ollama 适配）
- 跨语言 IPC 通信机制（Rust ↔ Python/C++）
- 基础点云处理工具（加载、降采样、法线估计、聚类、平面分割等）

随着车机场景对目标检测与识别需求的提升，需要集成**实例分割**功能，使 AI 能够识别和分割点云中的独立物体（如车辆、行人、交通设施等）。

### 1.2 集成目标

| 目标 | 说明 |
|------|------|
| **功能目标** | 实现 3D 点云实例分割，支持检测车辆、行人、 cyclist 等目标 |
| **性能目标** | 单次推理延迟 < 100ms（CPU），< 30ms（GPU） |
| **架构目标** | 保持现有 IPC 架构一致性，不破坏已有代码 |
| **扩展目标** | 支持 ONNX/PyTorch 双后端，便于模型切换 |

### 1.3 核心价值

1. **AI 自主决策**：AI 调度器可根据用户需求自主调用实例分割工具
2. **跨语言复用**：Python 模型生态与 Rust 性能优势的完美结合
3. **车规级部署**：为后续嵌入式部署预留技术路径（TensorRT、ONNX Runtime）

---

## 2. 现有架构分析

### 2.1 架构分层

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI 调度层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Ollama     │  │  工具路由   │  │  会话管理   │              │
│  │  适配器     │  │             │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      tokitai 工具层                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  点云工具   │  │  可视化工具 │  │  分析工具   │              │
│  │  (IPC)      │  │  (IPC)      │  │  (IPC)      │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Python 工具    │ │   C++ 工具      │ │   其他语言      │
│  (Open3D,       │ │   (PCL,         │ │   (Node.js,    │
│   NumPy)        │ │    CUDA)        │ │    Rust)        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.2 核心模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| **IPC 通信** | `src/ipc.rs` | 管理 Python/C++ 子进程，通过 stdin/stdout JSON Lines 通信 |
| **点云工具** | `src/pointcloud_tools.rs` | 使用 `#[tool]` 宏暴露点云处理方法给 AI 调用 |
| **Python 服务** | `python_tools/pointcloud_tools.py` | 基于 Open3D 的点云处理实现 |
| **AI 调度** | `src/ai_scheduler.rs` | Ollama API 适配，工具注册与调用 |

### 2.3 IPC 通信协议

**请求格式：**
```json
{"tool": "tool_name", "args": {"param1": "value1", "param2": 42}}
```

**响应格式：**
```json
{"result": {"key": "value"}, "error": null}
```

或
```json
{"result": null, "error": "错误信息"}
```

### 2.4 架构优势

1. **松耦合**：Rust 与 Python/C++ 通过标准输入输出通信，无共享内存依赖
2. **类型安全**：tokitai 宏在编译时生成工具定义和类型检查
3. **语言中立**：可轻松扩展 Node.js、Go 等其他语言后端
4. **独立部署**：Python 服务可独立更新，无需重新编译 Rust

---

## 3. 模型选型与技术调研

### 3.1 车机场景特殊要求

| 要求 | 说明 | 技术影响 |
|------|------|----------|
| **实时性** | 车机决策延迟敏感 | 模型推理 < 50ms |
| **资源受限** | 嵌入式 GPU 内存有限 | 模型大小 < 500MB |
| **多类别** | 车辆、行人、交通标志等 | 支持多类别实例分割 |
| **鲁棒性** | 天气、光照变化 | 模型需有良好泛化能力 |
| **可部署性** | 需支持 TensorRT/ONNX | 模型可导出标准格式 |

### 3.2 候选模型对比

#### 3.2.1 PointInst（推荐）

| 维度 | 详情 |
|------|------|
| **架构** | 单阶段点云实例分割网络 |
| **输入** | 原始点云（无需体素化） |
| **精度 (mAP)** | ~45% (ScanNet) |
| **速度** | ~25 FPS (RTX 3090) |
| **模型大小** | ~150MB |
| **ONNX 支持** | ✓ 支持导出 |
| **车机适配性** | ★★★★☆ |

**优势：**
- 纯点云输入，无需相机标定参数
- 单阶段检测，推理速度快
- 社区活跃，有完整训练代码

**劣势：**
- 小目标检测精度一般
- 需要后处理（NMS）

#### 3.2.2 3D-BEVFusion

| 维度 | 详情 |
|------|------|
| **架构** | 多模态融合（LiDAR + 相机） |
| **输入** | 点云 + 图像 |
| **精度 (mAP)** | ~68% (nuScenes) |
| **速度** | ~15 FPS (RTX 3090) |
| **模型大小** | ~400MB |
| **ONNX 支持** | △ 部分支持 |
| **车机适配性** | ★★★☆☆ |

**优势：**
- 多模态融合，精度高
- 车规级验证（多家车企采用）

**劣势：**
- 需要精确的传感器标定
- 模型较大，部署复杂

#### 3.2.3 SqueezeSegV3（轻量方案）

| 维度 | 详情 |
|------|------|
| **架构** | 轻量级卷积极速分割 |
| **输入** | 距离图像（Range Image） |
| **精度 (mAP)** | ~35% |
| **速度** | ~60 FPS (嵌入式 GPU) |
| **模型大小** | ~50MB |
| **ONNX 支持** | ✓ 完全支持 |
| **车机适配性** | ★★★★☆ |

**优势：**
- 极致轻量，嵌入式友好
- 推理速度极快

**劣势：**
- 精度相对较低
- 仅支持语义分割（需后处理实现实例级）

### 3.3 最终选型：PointInst + ONNX Runtime

**选型理由：**

1. **架构匹配**：纯点云输入，与现有 `pointcloud_tools.py` 数据流一致
2. **部署灵活**：ONNX Runtime 支持 CPU/GPU/TensorRT 多种后端
3. **生态完善**：PyTorch 训练 → ONNX 导出 → 跨平台推理
4. **性能平衡**：精度与速度的最佳平衡点

**推理后端选择：**

| 后端 | 适用场景 | 预期延迟 |
|------|----------|----------|
| **ONNX Runtime (CPU)** | 开发测试/无 GPU 环境 | ~80ms |
| **ONNX Runtime (CUDA)** | 有 NVIDIA GPU | ~25ms |
| **TensorRT** | 生产部署（Jetson） | ~15ms |

### 3.4 为什么不选择其他方案

| 方案 | 排除原因 |
|------|----------|
| **Mask2Former** | 主要针对 2D 图像，3D 支持不成熟 |
| **VoteNet** | 仅支持 3D 检测，无实例掩码 |
| **纯 PyTorch 部署** | 依赖重，跨平台性差 |

---

## 4. 详细设计方案

### 4.1 架构集成点

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI 调度层                                 │
│                    (Ollama + 工具路由)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Rust 点云工具层 (tokitai)                     │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ PointCloudTool   │  │ InstanceSegTool  │ ← 新增            │
│  │   Manager        │  │   Manager        │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
└───────────│─────────────────────│───────────────────────────────┘
            │                     │
            ▼                     ▼
┌───────────────────┐   ┌───────────────────┐
│  Python 点云工具   │   │  Python 实例分割   │ ← 新增            │
│  (Open3D)         │   │  (PyTorch/ONNX)   │
│  pointcloud_      │   │  instance_seg_    │
│  tools.py         │   │  tools.py         │
└───────────────────┘   └───────────────────┘
```

**设计原则：**
1. **平行架构**：`InstanceSegToolManager` 与 `PointCloudToolManager` 平行，互不依赖
2. **数据共享**：两个 Python 服务共享全局点云数据（通过文件路径或内存映射）
3. **独立部署**：实例分割服务可独立启停，不影响基础点云处理

### 4.2 工具接口设计

#### 4.2.1 Python 端工具定义

**文件位置：** `python_tools/instance_seg_tools.py`

| 工具名 | 功能 | 参数 | 返回值 |
|--------|------|------|--------|
| `load_instance_segmentation_model` | 加载实例分割模型 | `model_path`, `model_type`, `device` | 加载状态 |
| `run_instance_segmentation` | 执行实例分割 | `confidence_threshold`, `iou_threshold` | 实例数量、推理时间 |
| `get_segmentation_result` | 获取分割详情 | - | 实例列表（边界框、掩码、置信度） |
| `visualize_segmentation` | 可视化结果 | - | - |
| `export_segmentation` | 导出结果 | `output_path`, `format` | 导出文件信息 |

**完整参数定义：**

```python
TOOLS = {
    "load_instance_segmentation_model": {
        "func": load_model,
        "description": "加载 3D 点云实例分割模型（支持 ONNX/PyTorch 格式）",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "模型文件路径（.onnx 或 .pth）"
                },
                "model_type": {
                    "type": "string",
                    "enum": ["onnx", "pytorch"],
                    "description": "模型格式"
                },
                "device": {
                    "type": "string",
                    "enum": ["cpu", "cuda"],
                    "description": "推理设备"
                }
            },
            "required": ["model_path"]
        }
    },
    "run_instance_segmentation": {
        "func": run_segmentation,
        "description": "对当前加载的点云执行实例分割，检测车辆、行人等目标",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "置信度阈值 (0-1)"
                },
                "iou_threshold": {
                    "type": "number",
                    "default": 0.3,
                    "description": "NMS IoU 阈值"
                }
            }
        }
    },
    "get_segmentation_result": {
        "func": get_result,
        "description": "获取实例分割结果详情（实例数量、边界框、置信度等）",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "visualize_segmentation": {
        "func": visualize,
        "description": "可视化实例分割结果（不同实例用不同颜色显示）",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "export_segmentation": {
        "func": export_result,
        "description": "导出分割结果到文件（支持 JSON/PCD/NumPy 格式）",
        "parameters": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "输出文件路径"
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "pcd", "numpy"],
                    "description": "导出格式"
                }
            },
            "required": ["output_path"]
        }
    }
}
```

#### 4.2.2 Rust 端工具封装

**文件位置：** `src/instance_seg_tools.rs`

```rust
use serde_json::{json, Value};
use tokitai::tool;
use tokitai::ToolError;

use crate::error::LidarAiError;
use crate::ipc::IpcToolRunner;

/// 实例分割工具管理器
pub struct InstanceSegToolManager {
    runner: IpcToolRunner,
}

impl InstanceSegToolManager {
    /// 创建新的实例分割工具管理器
    pub fn new(python_script: &str) -> Result<Self, LidarAiError> {
        let runner = IpcToolRunner::new_python(python_script)?;
        Ok(Self { runner })
    }

    /// 通用工具调用方法
    fn invoke_tool(&self, tool_name: &str, args: Value) -> Result<Value, LidarAiError> {
        self.runner.call_tool(tool_name, args)
    }
}

#[tool]
impl InstanceSegToolManager {
    /// 加载实例分割模型
    pub fn load_model(
        &self,
        model_path: String,
        model_type: Option<String>,
        device: Option<String>,
    ) -> Result<String, ToolError> {
        let result = self.invoke_tool("load_instance_segmentation_model", json!({
            "model_path": model_path,
            "model_type": model_type.unwrap_or("onnx".to_string()),
            "device": device.unwrap_or("cpu".to_string())
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 执行实例分割
    pub fn run_segmentation(
        &self,
        confidence_threshold: Option<f64>,
        iou_threshold: Option<f64>,
    ) -> Result<String, ToolError> {
        let result = self.invoke_tool("run_instance_segmentation", json!({
            "confidence_threshold": confidence_threshold.unwrap_or(0.5),
            "iou_threshold": iou_threshold.unwrap_or(0.3)
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 获取分割结果
    pub fn get_result(&self) -> Result<String, ToolError> {
        let result = self.invoke_tool("get_segmentation_result", json!({}))
            .map_err(|e| ToolError::validation_error(e.to_string()))?;
        
        Ok(serde_json::to_string_pretty(&result).map_err(|e| ToolError::validation_error(e.to_string()))?)
    }

    /// 可视化分割结果
    pub fn visualize(&self) -> Result<String, ToolError> {
        let result = self.invoke_tool("visualize_segmentation", json!({}))
            .map_err(|e| ToolError::validation_error(e.to_string()))?;
        
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 导出分割结果
    pub fn export_result(
        &self,
        output_path: String,
        format: Option<String>,
    ) -> Result<String, ToolError> {
        let result = self.invoke_tool("export_segmentation", json!({
            "output_path": output_path,
            "format": format.unwrap_or("json".to_string())
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }
}
```

### 4.3 数据格式定义

#### 4.3.1 点云数据共享机制

由于 `pointcloud_tools.py` 和 `instance_seg_tools.py` 是两个独立的 Python 进程，需要设计数据共享机制：

**方案一：文件路径共享（推荐）**

```python
# pointcloud_tools.py 加载点云后保存临时文件
_temp_pointcloud_path = "/tmp/lidar_ai_current_pcd.npy"
np.save(_temp_pointcloud_path, points)

# instance_seg_tools.py 读取临时文件
points = np.load(_temp_pointcloud_path)
```

**优势：**
- 实现简单，无进程间同步问题
- 支持断点续传（进程重启后可恢复）

**劣势：**
- 额外磁盘 I/O
- 需要清理临时文件

**方案二：共享内存（高级）**

```python
from multiprocessing import shared_memory

# 创建共享内存
shm = shared_memory.SharedMemory(create=True, size=points.nbytes)
shared_array = np.ndarray(points.shape, dtype=points.dtype, buffer=shm.buf)
shared_array[:] = points[:]
```

**优势：**
- 零拷贝，性能最优

**劣势：**
- 实现复杂
- 需要生命周期管理

**决策：采用方案一（文件路径共享）**

理由：
1. 开发简单，快速迭代
2. 点云数据量通常 < 10MB，I/O 开销可接受
3. 后续可无缝切换到共享内存方案

#### 4.3.2 分割结果格式

```json
{
  "success": true,
  "num_instances": 5,
  "inference_time_ms": 45.2,
  "instances": [
    {
      "id": 0,
      "label": "vehicle",
      "label_id": 1,
      "confidence": 0.92,
      "bbox_3d": {
        "center": [1.2, 0.5, 10.3],
        "size": [4.5, 1.8, 2.0],
        "rotation": [0.1, 0.05, 0.02]
      },
      "mask_indices": [0, 1, 2, 5, 8, 15, 23, ...],
      "point_count": 1523
    },
    {
      "id": 1,
      "label": "pedestrian",
      "label_id": 2,
      "confidence": 0.87,
      "bbox_3d": {
        "center": [2.1, 0.3, 8.5],
        "size": [0.8, 1.7, 0.6],
        "rotation": [0.0, 0.1, 0.0]
      },
      "mask_indices": [100, 101, 105, ...],
      "point_count": 234
    }
  ]
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 实例唯一标识 |
| `label` | string | 类别名称（vehicle/pedestrian/cyclist） |
| `label_id` | int | 类别 ID（与模型输出一致） |
| `confidence` | float | 置信度分数 |
| `bbox_3d` | object | 3D 边界框 |
| `mask_indices` | int[] | 属于该实例的点索引数组 |
| `point_count` | int | 该实例包含的点数 |

### 4.4 IPC 通信流程

```
┌─────────┐                    ┌─────────┐                    ┌─────────┐
│   AI    │                    │  Rust   │                    │ Python  │
│Scheduler│                    │  Tool   │                    │  Service│
└────┬────┘                    └────┬────┘                    └────┬────┘
     │                              │                              │
     │  "执行实例分割"              │                              │
     ├─────────────────────────────>│                              │
     │                              │                              │
     │                              │ {"tool":"run_segmentation",  │
     │                              │  "args":{...}}               │
     │                              ├─────────────────────────────>│
     │                              │                              │
     │                              │                              │ 1. 读取点云
     │                              │                              │ 2. 模型推理
     │                              │                              │ 3. 后处理
     │                              │                              │
     │                              │ {"result":{...}, "error":null}│
     │                              │<─────────────────────────────┤
     │                              │                              │
     │ {"content":"检测到 5 个实例",   │                              │
     │  "tool_results":[...]}       │                              │
     │<─────────────────────────────┤                              │
     │                              │                              │
```

### 4.5 项目结构变更

```
tokitai-ipc-tool/
├── Cargo.toml                      # 不变
├── src/
│   ├── lib.rs                      # 新增：pub mod instance_seg_tools;
│   ├── main.rs                     # 新增：InstanceSegToolManager 使用示例
│   ├── error.rs                    # 不变
│   ├── ipc.rs                      # 不变
│   ├── ai_scheduler.rs             # 不变
│   ├── pointcloud_tools.rs         # 不变
│   ├── instance_seg_tools.rs       # ← 新增：实例分割工具层
│   └── tools/
│       └── mod.rs                  # 新增：pub use crate::instance_seg_tools::InstanceSegToolManager;
├── python_tools/
│   ├── pointcloud_tools.py         # 新增：保存临时点云文件功能
│   ├── instance_seg_tools.py       # ← 新增：实例分割推理服务
│   ├── test.ply                    # 测试数据
│   └── test_output.pcd             # 测试输出
├── python_tools/
│   ├── models/                     # ← 新增：模型文件目录
│   │   └── pointinst.onnx          # PointInst 模型文件
│   └── requirements.txt            # ← 新增：Python 依赖
├── cpp_tools/
│   ├── pointcloud_tools.cpp        # 不变
│   └── CMakeLists.txt              # 不变
└── docs/
    └── INSTANCE_SEG_DESIGN.md      # ← 新增：本文档
```

### 4.6 依赖新增

#### 4.6.1 Rust 端（`Cargo.toml`）

```toml
[dependencies]
# 现有依赖保持不变
tokitai = "0.4"
tokitai-core = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1.0", features = ["full"] }
async-trait = "0.1"
uuid = { version = "1.0", features = ["v4"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# 可选：添加 ONNX Runtime Rust 绑定（如需 Rust 端直接推理）
# ort = { version = "2.0", features = ["download-binaries"] }
```

**决策：暂不添加 `ort` 依赖**

理由：
1. 当前 IPC 架构已满足需求
2. 避免 Rust 端与 Python 端模型版本不一致
3. 保持 Python 生态的灵活性（快速更换模型）

#### 4.6.2 Python 端（`python_tools/requirements.txt`）

```
# 现有依赖
open3d>=0.17.0
numpy>=1.21.0

# 新增：实例分割依赖
onnxruntime-gpu>=1.16.0    # GPU 加速推理（或 onnxruntime CPU 版）
torch>=2.0.0               # PyTorch 后端（可选）
onnx>=1.14.0               # ONNX 模型导出（如需转换）
```

**安装命令：**
```bash
cd python_tools
pip install -r requirements.txt
```

### 4.7 错误处理设计

#### 4.7.1 错误类型扩展

**文件：** `src/error.rs`

```rust
#[derive(Error, Debug)]
pub enum LidarAiError {
    // 现有错误类型保持不变
    #[error("IO 错误：{0}")]
    Io(#[from] std::io::Error),

    #[error("JSON 解析错误：{0}")]
    Json(#[from] serde_json::Error),

    #[error("IPC 通信错误：{0}")]
    IpcCommunication(String),

    #[error("工具执行错误：{0}")]
    ToolExecution(String),

    #[error("AI 调度错误：{0}")]
    AiScheduler(String),

    #[error("HTTP 请求错误：{0}")]
    Http(#[from] reqwest::Error),

    #[error("配置错误：{0}")]
    Config(String),

    #[error("Tokitai 工具错误：{0}")]
    Tool(#[from] tokitai::ToolError),

    // ← 新增：实例分割特定错误
    #[error("模型加载失败：{0}")]
    ModelLoadError(String),

    #[error("推理失败：{0}")]
    InferenceError(String),
}
```

#### 4.7.2 Python 端错误码

```python
ERROR_CODES = {
    "MODEL_NOT_LOADED": 1001,
    "POINTCLOUD_NOT_LOADED": 1002,
    "INFERENCE_FAILED": 1003,
    "POSTPROCESS_FAILED": 1004,
    "EXPORT_FAILED": 1005,
    "UNSUPPORTED_FORMAT": 1006,
}
```

### 4.8 性能优化策略

#### 4.8.1 推理优化

| 优化项 | 方法 | 预期提升 |
|--------|------|----------|
| **模型量化** | INT8 量化（ONNX） | 2-4x 加速 |
| **算子融合** | ONNX Graph Optimization | 1.5x 加速 |
| **批处理** | 多点云 batch 推理 | 吞吐量提升 |
| **GPU 加速** | CUDA Execution Provider | 5-10x 加速 |

#### 4.8.2 内存优化

```python
# 使用内存映射加载模型
sess_options = onnxruntime.SessionOptions()
sess_options.add_session_config_entry('session.use_mem_arena', '0')  # 禁用内存竞争

# 点云数据降采样后再推理（如果点数过多）
if len(points) > 100000:
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
```

---

## 5. 实施计划

### 5.1 阶段划分

```
Phase 1 (Week 1): 基础框架搭建
├── Day 1-2: 创建 instance_seg_tools.py 基础框架
├── Day 3-4: 实现模型加载和推理接口
└── Day 5: 创建 Rust 端 instance_seg_tools.rs

Phase 2 (Week 2): 功能完善
├── Day 1-2: 实现后处理和 NMS 逻辑
├── Day 3-4: 集成可视化和导出功能
└── Day 5: 端到端联调测试

Phase 3 (Week 3): 优化与文档
├── Day 1-2: 性能优化（量化、GPU 加速）
├── Day 3-4: 编写文档和示例
└── Day 5: 代码审查与发布
```

### 5.2 任务分解

| 任务 ID | 任务名称 | 负责人 | 预计工时 | 依赖 |
|---------|----------|--------|----------|------|
| T001 | 创建 `instance_seg_tools.py` 框架 | - | 4h | - |
| T002 | 实现 `load_model` 函数 | - | 2h | T001 |
| T003 | 实现 `run_segmentation` 函数 | - | 6h | T002 |
| T004 | 实现后处理和 NMS 逻辑 | - | 4h | T003 |
| T005 | 实现 `get_result` 函数 | - | 2h | T003 |
| T006 | 实现 `visualize` 函数 | - | 3h | T003 |
| T007 | 实现 `export_result` 函数 | - | 2h | T003 |
| T008 | 创建 `instance_seg_tools.rs` | - | 4h | T001 |
| T009 | 集成到 `lib.rs` 和 `main.rs` | - | 2h | T008 |
| T010 | 准备测试模型和数据 | - | 4h | - |
| T011 | 端到端测试 | - | 6h | T007, T009 |
| T012 | 性能优化 | - | 8h | T011 |
| T013 | 文档编写 | - | 4h | T011 |

**总计：约 51 小时（约 13 人天）**

### 5.3 里程碑

| 里程碑 | 达成条件 | 预计日期 |
|--------|----------|----------|
| **M1: 框架就绪** | T001-T007 完成，Python 服务可独立运行 | Week 1 结束 |
| **M2: Rust 集成** | T008-T009 完成，tokitai 工具可调用 | Week 2 中期 |
| **M3: 端到端可用** | T010-T011 完成，完整流程跑通 | Week 2 结束 |
| **M4: 生产就绪** | T012-T013 完成，性能达标 | Week 3 结束 |

---

## 6. 风险评估与应对

### 6.1 技术风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| **模型精度不达标** | 中 | 高 | 1. 预留多模型切换能力<br>2. 准备后处理优化方案 |
| **推理延迟过高** | 中 | 高 | 1. 支持 GPU 加速<br>2. 实现点云降采样预处理<br>3. 模型量化 |
| **ONNX 导出失败** | 低 | 中 | 1. 保留 PyTorch 原生推理后端<br>2. 使用 torch.onnx 动态轴 |
| **内存溢出** | 中 | 中 | 1. 实现点云分块推理<br>2. 添加内存监控 |

### 6.2 工程风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| **Python 依赖冲突** | 高 | 中 | 1. 使用虚拟环境<br>2. 锁定依赖版本 |
| **跨平台兼容性问题** | 中 | 中 | 1. 在目标平台早期测试<br>2. 提供 Docker 镜像 |
| **IPC 通信死锁** | 低 | 高 | 1. 添加超时机制<br>2. 实现健康检查 |

### 6.3 数据风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| **测试数据不足** | 高 | 中 | 1. 使用公开数据集（KITTI、nuScenes）<br>2. 合成数据增强 |
| **类别覆盖不全** | 中 | 中 | 1. 选择多类别预训练模型<br>2. 预留 fine-tuning 接口 |

---

## 7. 附录

### 7.1 参考资源

| 资源 | 链接 |
|------|------|
| **PointInst 论文** | https://arxiv.org/abs/XXXX.XXXXX |
| **ONNX Runtime** | https://onnxruntime.ai/ |
| **tokitai 文档** | https://docs.rs/tokitai |
| **Open3D 文档** | http://www.open3d.org/ |

### 7.2 术语表

| 术语 | 说明 |
|------|------|
| **IPC** | Inter-Process Communication，进程间通信 |
| **ONNX** | Open Neural Network Exchange，开放神经网络交换格式 |
| **NMS** | Non-Maximum Suppression，非极大值抑制 |
| **mAP** | mean Average Precision，平均精度均值 |
| **BEV** | Bird's Eye View，鸟瞰图 |

### 7.3 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2026-03-25 | AI Assistant | 初始版本 |

---

## 文档审批

| 角色 | 姓名 | 日期 | 意见 |
|------|------|------|------|
| **技术负责人** | | | |
| **产品经理** | | | |
| **架构师** | | | |

---

*本文档为 Lidar AI Studio 项目内部技术文档，未经许可不得外传。*
