# Lidar AI Studio - 完整项目文档

**版本**: 0.1.0
**最后更新**: 2026 年 4 月 8 日
**项目根目录**: `/home/hugo/codes/Ponai/tokitai-ipc-tool`

**实现状态**: ✅ 核心功能已完成
- ✅ IPC/HTTP 双后端切换机制
- ✅ 点云处理工具（Open3D）
- ✅ 实例分割工具（ONNX Runtime）
- ✅ PointPillars 3D 检测模型
- ✅ AI 调度器（Ollama 适配）
- ✅ 35+ 种 IPC 错误码
- ✅ 44+ Rust 单元测试

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [模块详解](#3-模块详解)
4. [数据流向](#4-数据流向)
5. [AI 模型清单](#5-ai-模型清单)
6. [工具 API 参考](#6-工具 API 参考)
7. [技术栈与依赖](#7-技术栈与依赖)
8. [开发指南](#8-开发指南)

---

## 1. 项目概述

### 1.1 项目定位

**Lidar AI Studio** 是一个面向 3D 点云车机应用的 AI 调度与跨语言工具框架。核心目标：

- 🎯 **AI 驱动**: 通过本地 LLM (Ollama) 实现自然语言到工具调用的转换
- 🔗 **跨语言 IPC**: Rust 调度层通过 JSON Lines IPC 调用 Python/C++ 工具
- 🚀 **双后端支持**: 支持本地 IPC 和远程 HTTP 两种后端模式
- 🧠 **点云处理**: 完整的点云处理工具链 (加载、滤波、分割、聚类、检测)

### 1.2 核心特性

| 特性 | 说明 | 状态 |
|------|------|------|
| **tokitai 宏** | 使用 `#[tool]` 宏自动暴露 Rust 方法给 AI 调度器 | ✅ |
| **IPC 通信** | stdin/stdout JSON Lines 协议，轻量级跨语言调用 | ✅ |
| **BackendSwitch** | 运行时动态切换 IPC/HTTP 后端 | ✅ |
| **Ollama 适配** | 支持本地 LLM 进行工具路由和任务规划 | ✅ |
| **多模型支持** | ONNX/PyTorch 模型推理 (PointPillars, PointInst) | ✅ |
| **错误处理** | 35+ 种 IPC 错误码，结构化错误通信 | ✅ |
| **路径解析** | PathResolver 支持运行时路径解析 | ✅ |
| **日志追踪** | tracing span + debug 级别日志 | ✅ |

### 1.3 应用场景

- 🚗 **自动驾驶**: 3D 点云目标检测、实例分割
- 🗺️ **高精地图**: 点云处理、地面分割、特征提取
- 🔍 **工业检测**: 3D 扫描数据分析、缺陷检测
- 📊 **科研开发**: 点云算法原型验证、模型推理测试

### 1.4 模型性能对比

| 模型 | 参数量 | 模型大小 | CPU FPS | GPU FPS | 推荐场景 |
|------|--------|----------|---------|---------|----------|
| **PointPillars** ⭐ | 5.2M | 15 MB | 18.5 | 95 | 实时车机应用 |
| SECOND | 8.3M | 24 MB | 14.0 | 58 | 精度优先 |
| PointInst | 12.5M | 35 MB | 15.0 | 72 | 实例分割 |
| CenterPoint | 15.8M | 45 MB | 12.0 | 65 | 多类别检测 |

详细对比报告：[`MODEL_COMPARISON_REPORT.md`](./MODEL_COMPARISON_REPORT.md)

---

## 2. 系统架构

### 2.1 三层架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Layer 1: AI 调度层                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   Ollama        │  │   AiScheduler   │  │   会话管理      │          │
│  │   (LLM 服务)    │◄─┤   (API 适配器)  │◄─┤   (历史/工具)   │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Layer 2: tokitai 工具层                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  PointCloudToolManager        InstanceSegToolManager              │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                  BackendSwitch (路由核心)                    │  │  │
│  │  │  ┌──────────────────┐  ┌──────────────────┐                 │  │  │
│  │  │  │  IPC Runner      │  │  HTTP Client     │                 │  │  │
│  │  │  │  (stdin/stdout)  │  │  (REST API)      │                 │  │  │
│  │  │  └──────────────────┘  └──────────────────┘                 │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ IPC 模式                      │ HTTP 模式
                    ▼                               ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│   Python IPC Services   │           │   Python HTTP Server    │
│   (stdin/stdout JSON)   │           │   (FastAPI REST API)    │
│                         │           │                         │
│  - pointcloud_tools.py  │           │  - instance_seg_server  │
│  - instance_seg_tools.py│           │  - /api/v1/* endpoints  │
│  - pointpillars_tools.py│           │                         │
└─────────────────────────┘           └─────────────────────────┘
            │                                     │
            ▼                                     ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│   Open3D / NumPy        │           │   ONNX Runtime          │
│   (CPU 点云处理)         │           │   (CPU/GPU 推理)        │
└─────────────────────────┘           └─────────────────────────┘
```

**架构演进**:
- **v0.1.0** (2026-04-08): 完成 IPC/HTTP 双后端切换，重构锁架构
- **v0.0.3** (2026-04-03): 添加 HTTP 服务支持
- **v0.0.2** (2026-04-02): 添加实例分割工具
- **v0.0.1** (2026-04-01): 初始版本，仅 IPC 点云处理

### 2.2 组件职责

| 层级 | 组件 | 职责 |
|------|------|------|
| **AI 调度层** | Ollama | 自然语言理解、工具选择、任务规划 |
| | AiScheduler | Ollama API 封装、会话管理、工具注册 |
| **工具层** | PointCloudToolManager | 点云处理工具包装 (Open3D) |
| | InstanceSegToolManager | 实例分割工具包装 (ONNX/PyTorch) |
| | BackendSwitch | IPC/HTTP 路由、动态切换 |
| **执行层** | Python IPC | 点云处理、模型推理 |
| | Python HTTP | 远程 GPU 推理服务 |
| | C++ Tools | 高性能计算 (框架，待实现) |

---

## 3. 模块详解

### 3.1 Rust 模块 (`src/`)

#### 3.1.1 `lib.rs` - 库入口

**作用**: 导出公共 API

```rust
pub use ai_scheduler::{AiScheduler, AiSchedulerConfig};
pub use backend_switch::BackendSwitch;
pub use error::{LidarAiError, Result};
pub use instance_seg_tools::InstanceSegToolManager;
pub use pointcloud_tools::PointCloudToolManager;
```

#### 3.1.2 `error.rs` - 错误类型

**作用**: 统一错误处理，支持跨语言结构化错误通信

**核心特性**:
- 所有错误转换为 `LidarAiError`
- 支持从 IPC 结构化错误转换
- 提供 `is_recoverable()` 和 `is_server_error()` 方法

```rust
pub enum LidarAiError {
    Io(#[from] std::io::Error),           // IO 错误
    Json(#[from] serde_json::Error),      // JSON 解析错误
    IpcCommunication(String),             // IPC 通信错误
    ToolExecution {                       // 工具执行错误（带错误码）
        code: ErrorCode,
        message: String,
    },
    AiScheduler(String),                  // AI 调度错误
    Http(String),                         // HTTP 请求错误
    Config(String),                       // 配置错误
    Tool(#[from] tokitai::ToolError),     // tokitai 工具错误
}
```

**错误分类**:
| 错误类型 | 可恢复性 | 服务端责任 |
|----------|----------|------------|
| `CommunicationTimeout` | ✅ 可恢复 | ✅ 是 |
| `ConnectionLost` | ✅ 可恢复 | ✅ 是 |
| `ServiceUnavailable` | ✅ 可恢复 | ✅ 是 |
| `RateLimitExceeded` | ✅ 可恢复 | ❌ 否 |
| `OutOfMemory` | ❌ 不可恢复 | ✅ 是 |
| `LockPoisoned` | ❌ 不可恢复 | ✅ 是 |
| `InvalidParameter` | ❌ 不可恢复 | ❌ 否 |
| `FileNotFound` | ❌ 不可恢复 | ❌ 否 |

**IO/HTTP 错误处理**:
- IO 超时/连接断开：可恢复
- HTTP 超时/503：可恢复
- IO 其他错误：不可恢复
- HTTP 4xx：客户端错误

#### 3.1.3 `ipc.rs` - IPC 通信

**作用**: 管理外部进程，提供 stdin/stdout JSON 通信

**核心结构**:
- `IpcRequest`: `{"tool": String, "args": Value}`
- `IpcResponse`: `{"result": Option<Value>, "error": Option<String>}`
- `IpcToolRunner`: 进程管理 + 工具调用

**通信协议**:
```
Rust → Python: {"tool": "load_point_cloud", "args": {"file_path": "/data.pcd"}}\n
Python → Rust: {"result": {"message": "..."}, "error": null}\n
```

#### 3.1.4 `backend_switch.rs` - 后端切换器

**作用**: 支持 IPC/HTTP 双后端，运行时动态切换

**架构演进**:
- **旧架构**: `Arc<Mutex<Arc<Mutex<T>>>>` 嵌套锁（已废弃）
- **新架构**: `Arc<RwLock<Box<dyn Backend>>>` 单锁，读写分离

**核心类型**:
```rust
pub enum BackendType {
    Ipc,   // 本地进程调用
    Http,  // 网络 HTTP 调用
}

pub struct BackendSwitch {
    inner: Arc<RwLock<BackendSwitchInner>>,
}

struct BackendSwitchInner {
    backend_type: BackendType,
    backend: Box<dyn Backend>,
}
```

**关键方法**:
| 方法 | 说明 | 锁类型 |
|------|------|--------|
| `new_ipc(script_path)` | 创建 IPC 后端 | - |
| `new_http(base_url, api_key, timeout)` | 创建 HTTP 后端 | - |
| `switch_to_ipc(script_path)` | 动态切换到 IPC | 写锁 |
| `switch_to_http(base_url, ...)` | 动态切换到 HTTP | 写锁 |
| `current_backend()` | 获取当前后端类型 | 读锁 |
| `call_tool(tool_name, args)` | 统一工具调用 (自动路由) | 读锁 |

**并发特性**:
- 读操作（`call_tool`）：共享锁，支持并发
- 写操作（`switch_to_*`）：独占锁，阻塞其他操作
- Clone 语义：多个 `BackendSwitch` clone 共享同一个后端实例

#### 3.1.5 `ai_scheduler.rs` - AI 调度器

**作用**: Ollama API 适配器，管理会话和工具调用

**配置结构**:
```rust
pub struct AiSchedulerConfig {
    pub host: String,      // "localhost"
    pub port: u16,         // 11434
    pub model: String,     // "llama3.2"
    pub stream: bool,      // false
}
```

**核心方法**:
| 方法 | 说明 |
|------|------|
| `register_tools(tools)` | 注册工具定义到 AI |
| `chat(user_message)` | 发送消息给 Ollama |
| `process_tool_result(tool_call, result)` | 处理工具调用结果 |
| `clear_history()` | 清除会话历史 |

#### 3.1.6 `pointcloud_tools.rs` - 点云工具管理器

**作用**: 使用 `#[tool]` 宏包装 Python 点云工具

**暴露的工具方法**:
| 方法 | 后端 | 说明 | 参数 |
|------|------|------|------|
| `load_point_cloud(file_path)` | Python | 加载点云文件 | file_path: str |
| `get_point_cloud_info()` | Python | 获取点云信息 | - |
| `downsample(voxel_size)` | Python | 体素降采样 | voxel_size: f64 |
| `estimate_normals(k_neighbors)` | Python | 法线估计 | k_neighbors: u32 |
| `remove_outliers(nb_neighbors, std_ratio)` | Python | 离群点移除 | nb_neighbors: u32, std_ratio: f64 |
| `segment_plane(distance_threshold, max_iterations)` | Python | 平面分割 | distance_threshold: f64, max_iterations: u32 |
| `euclidean_clustering(...)` | Python | 欧式聚类 | tolerance, min/max_cluster_size |
| `save_point_cloud(file_path)` | Python | 保存点云 | file_path: str |
| `visualize()` | Python | 可视化 | - |
| `get_cpp_tools()` | C++ | 获取 C++ 工具列表 | - |

#### 3.1.7 `instance_seg_tools.rs` - 实例分割工具管理器

**作用**: 支持 IPC/HTTP 双后端的实例分割工具包装

**暴露的工具方法**:
| 方法 | 后端 | 说明 | 参数 |
|------|------|------|------|
| `load_model(model_path, model_type, device)` | 双后端 | 加载分割模型 | model_path: str, model_type: "onnx"/"pytorch", device: "cpu"/"cuda" |
| `run_segmentation(confidence_threshold, iou_threshold)` | 双后端 | 执行实例分割 | confidence_threshold: f64, iou_threshold: f64 |
| `get_result()` | 双后端 | 获取分割结果 | - |
| `visualize()` | 双后端 | 可视化结果 | - |
| `export_result(output_path, format)` | 双后端 | 导出结果 | output_path: str, format: "json"/"pcd"/"numpy" |
| `switch_backend(backend_type, config)` | - | 动态切换后端 | backend_type: "ipc"/"http", config: str |
| `get_backend_info()` | - | 获取后端信息 | - |

### 3.2 Python 模块 (`python_tools/`)

#### 3.2.1 `pointcloud_tools.py` - 点云处理服务

**作用**: 基于 Open3D 的点云处理工具实现

**依赖**:
- `open3d` - 点云 I/O 和处理
- `numpy` - 数值计算
- `laspy` (可选) - LAS 格式支持

**核心功能**:
| 函数 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `load_point_cloud(file_path)` | 加载 PCD/PLY/LAS | 文件路径 | 点云信息 |
| `get_point_cloud_info()` | 获取点云统计 | - | 点数、边界框、密度等 |
| `downsample(voxel_size)` | 体素降采样 | 体素大小 | 降采样统计 |
| `estimate_normals(k_neighbors)` | 法线估计 | 邻域数 | 法线信息 |
| `remove_outliers(...)` | 离群点移除 | 邻域数、标准差比 | 移除统计 |
| `segment_plane(...)` | RANSAC 平面分割 | 距离阈值、迭代次数 | 平面模型、内外点 |
| `euclidean_clustering(...)` | DBSCAN 聚类 | 容差、簇大小 | 簇列表 |
| `save_point_cloud(file_path)` | 保存点云 | 文件路径 | 保存结果 |
| `visualize()` | 打开可视化窗口 | - | - |

**数据共享机制**:
- 点云数据保存到 `tmp/lidar_ai_current_pcd.npy`
- 其他工具 (如 instance_seg_tools.py) 通过该文件共享点云

#### 3.2.2 `instance_seg_tools.py` - 实例分割服务

**作用**: 基于 ONNX Runtime / PyTorch 的 3D 点云实例分割

**依赖**:
- `onnxruntime` / `onnxruntime-gpu` - ONNX 推理
- `torch` (可选) - PyTorch 支持
- `open3d` - 可视化

**模型支持**:
- **ONNX**: 推荐，支持 CPU/GPU 推理
- **PyTorch**: 训练/调试用

**核心流程**:
```
1. load_model(model_path) → 加载 ONNX/PyTorch 模型
2. run_segmentation(conf_thresh, iou_thresh) → 推理
3. get_result() → 获取实例列表 (边界框、掩码、置信度)
4. visualize() → 彩色可视化
5. export_result(path, format) → 导出 JSON/PCD/Numpy
```

**输出结构**:
```json
{
  "num_instances": 3,
  "inference_time_ms": 45.2,
  "instances": [
    {
      "id": 0,
      "label": "vehicle",
      "confidence": 0.92,
      "bbox_3d": {
        "center": [x, y, z],
        "size": [l, w, h],
        "rotation": 0.0
      },
      "mask_indices": [1, 5, 23, ...]
    }
  ]
}
```

#### 3.2.3 `pointpillars_tools.py` - PointPillars 3D 检测

**作用**: PointPillars 3D 目标检测模型推理

**模型信息**:
- **架构**: Pillar 编码 + 2D CNN + 检测头
- **输入**: 点云 [1, 4096, 4] (x, y, z, intensity)
- **输出**: 边界框 [1, 7], 置信度 [1], 类别 [1]
- **类别**: Car, Pedestrian, Cyclist (KITTI 数据集)

**API**:
| 函数 | 说明 |
|------|------|
| `load_pointpillars(model_path, device)` | 加载 ONNX 模型 |
| `run_pointpillars(points, confidence_threshold)` | 执行推理 |
| `get_pointpillars_info()` | 获取模型信息 |

#### 3.2.4 `instance_seg_server.py` - HTTP 服务

**作用**: FastAPI 实现的 REST API 服务

**端点**:
| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/load_instance_segmentation_model` | POST | 加载模型 |
| `/api/v1/run_instance_segmentation` | POST | 执行分割 |
| `/api/v1/get_segmentation_result` | POST | 获取结果 |
| `/api/v1/visualize_segmentation` | POST | 可视化 |
| `/api/v1/export_segmentation` | POST | 导出结果 |
| `/api/v1/batch_segmentation` | POST | 批量分割 |

**特性**:
- API Key 认证 (`Authorization: Bearer <key>`)
- CORS 跨域支持
- 异步处理

#### 3.2.5 `start_server.sh` - 启动脚本

```bash
# 启动 HTTP 服务
./python_tools/start_server.sh --host 0.0.0.0 --port 8080 --api-key your-key

# 开发模式 (自动重载)
./python_tools/start_server.sh --reload
```

### 3.3 C++ 模块 (`cpp_tools/`)

#### 3.3.1 `pointcloud_tools.cpp` - C++ 工具框架

**作用**: 高性能点云处理 (框架，待实现)

**依赖**:
- `jsoncpp` - JSON 解析
- `PCL` (可选) - Point Cloud Library
- `CUDA` (可选) - GPU 加速

**计划工具**:
| 工具 | 说明 |
|------|------|
| `gpu_accelerated_filter` | GPU 加速滤波 |
| `real_time_segmentation` | 实时分割 |
| `cuda_normals` | CUDA 加速法线估计 |

#### 3.3.2 `CMakeLists.txt` - 构建配置

```bash
# 编译 C++ 工具
cd cpp_tools && mkdir -p build && cd build
cmake ..
make
# 输出：cpp_tools/build/pointcloud_tools_cpp
```

---

## 4. 数据流向

### 4.1 AI 工具调用流程

```
用户输入: "帮我加载点云并进行降采样"
    │
    ▼
┌─────────────────────────────────────┐
│  Ollama (LLM)                       │
│  1. 理解用户意图                     │
│  2. 选择工具: load_point_cloud      │
│  3. 提取参数：file_path="/data.pcd" │
└─────────────────────────────────────┘
    │
    │ Tool Call: {"name": "load_point_cloud", "args": {...}}
    ▼
┌─────────────────────────────────────┐
│  AiScheduler                        │
│  1. 接收 Tool Call                   │
│  2. 调用 PointCloudToolManager      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  PointCloudToolManager.load_point_cloud()
│  调用：invoke_tool("python", "load_point_cloud", args)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  IpcToolRunner.call_tool()          │
│  1. 写入 stdin: {"tool": "..."}     │
│  2. 读取 stdout: {"result": "..."}  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Python: pointcloud_tools.py        │
│  1. load_point_cloud(file_path)     │
│  2. 保存到 tmp/lidar_ai_current_pcd.npy
└─────────────────────────────────────┘
    │
    │ 结果返回
    ▼
┌─────────────────────────────────────┐
│  AiScheduler.process_tool_result()  │
│  1. 将结果发送给 Ollama              │
│  2. 获取 AI 的自然语言响应            │
└─────────────────────────────────────┘
    │
    ▼
用户收到："已成功加载点云文件，包含 10,234 个点"
```

### 4.2 后端切换流程

```
初始状态：IPC 后端
    │
    ▼
InstanceSegToolManager.new_ipc("python_tools/instance_seg_tools.py")
    │
    ▼
BackendSwitch 内部:
  - backend_type = Ipc
  - ipc_runner = Some(IpcToolRunner)
    │
    │ 用户调用：switch_backend("http", "http://gpu-server:8080")
    ▼
BackendSwitch.switch_to_http("http://gpu-server:8080", None, 30)
    │
    ▼
BackendSwitch 内部更新:
  - backend_type = Http
  - http_client = Some(reqwest::Client)
  - http_config = Some(HttpConfig { base_url: "...", ... })
    │
    │ 后续调用自动路由到 HTTP
    ▼
call_tool("run_segmentation", args)
    │
    ├─► if backend_type == Ipc: 调用 ipc_runner.call_tool()
    │
    └─► if backend_type == Http: 调用 Self::call_http_tool()
           │
           ▼
           POST http://gpu-server:8080/api/v1/run_segmentation
           Body: {"args": {...}}
```

### 4.3 点云数据共享流程

```
1. pointcloud_tools.py 加载点云
   load_point_cloud("/data.pcd")
   │
   ├─► Open3D: o3d.io.read_point_cloud()
   │
   └─► 保存到共享文件:
       np.save("tmp/lidar_ai_current_pcd.npy", points)

2. instance_seg_tools.py 读取点云
   run_segmentation()
   │
   └─► 从共享文件加载:
       points = np.load("tmp/lidar_ai_current_pcd.npy")
       │
       ▼
       模型推理 → 返回实例分割结果

3. pointpillars_tools.py 同样从共享文件读取
```

---

## 5. AI 模型清单

### 5.1 本地 LLM 模型

| 模型 | 用途 | 配置 |
|------|------|------|
| **llama3.2** | 默认 AI 调度模型 | Ollama: `localhost:11434` |

**启动 Ollama**:
```bash
# 安装
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull llama3.2

# 启动服务
ollama serve
```

### 5.2 点云处理模型

#### 5.2.1 PointPillars (3D 目标检测)

**位置**: `python_tools/models/`

| 模型文件 | 说明 | 格式 |
|----------|------|------|
| `pointpillars.onnx` | 默认模型 | ONNX |
| `pointpillars_simple.onnx` | 简化版 | ONNX |
| `pointpillars_realistic.onnx` | 真实场景版 | ONNX |
| `pointpillars_kitti_3class.pth` | KITTI 3 类检测 | PyTorch |

**模型规格**:
- **输入**: [1, 4096, 4] (batch, points, features)
- **输出**: 
  - boxes: [1, 7] (x, y, z, l, w, h, rotation)
  - scores: [1] (置信度)
  - labels: [1] (类别 ID)
- **类别**: Car (0), Pedestrian (1), Cyclist (2)

#### 5.2.2 PointInst (实例分割)

**位置**: `python_tools/models/` (需手动添加)

| 模型 | 说明 |
|------|------|
| `pointinst.onnx` | 实例分割模型 (待添加) |

**输出结构**:
```json
{
  "instances": [
    {
      "id": 0,
      "label": "vehicle",
      "confidence": 0.92,
      "bbox_3d": {"center": [...], "size": [...], "rotation": 0.0},
      "mask_indices": [1, 5, 23, ...]
    }
  ]
}
```

### 5.3 模型推理后端

| 后端 | 依赖 | 设备 | 性能 |
|------|------|------|------|
| **ONNX Runtime CPU** | `onnxruntime` | CPU | 中等 |
| **ONNX Runtime GPU** | `onnxruntime-gpu` | CUDA | 高 |
| **PyTorch** | `torch` | CPU/CUDA | 高 (训练用) |

---

## 6. 工具 API 参考

### 6.1 点云工具 (PointCloudToolManager)

#### `load_point_cloud(file_path: str) -> str`

加载点云文件 (支持 PCD, PLY, LAS 格式)

**参数**:
- `file_path`: 点云文件路径

**返回**: 成功消息

**示例**:
```rust
let result = manager.load_point_cloud("/data/scan.pcd".to_string())?;
println!("{}", result);  // "成功加载点云文件：/data/scan.pcd"
```

#### `downsample(voxel_size: f64) -> str`

体素网格降采样

**参数**:
- `voxel_size`: 体素大小 (建议 0.01-0.1)

**返回**: 降采样统计

#### `segment_plane(distance_threshold: f64, max_iterations: u32) -> str`

RANSAC 平面分割

**参数**:
- `distance_threshold`: 距离阈值 (建议 0.01-0.1)
- `max_iterations`: 最大迭代次数 (建议 1000-10000)

**返回**: 平面模型方程、内外点统计

### 6.2 实例分割工具 (InstanceSegToolManager)

#### `load_model(model_path: str, model_type: Option<String>, device: Option<String>) -> str`

加载实例分割模型

**参数**:
- `model_path`: 模型文件路径
- `model_type`: "onnx" 或 "pytorch" (默认 "onnx")
- `device`: "cpu" 或 "cuda" (默认 "cpu")

**返回**: 加载结果

#### `run_segmentation(confidence_threshold: Option<f64>, iou_threshold: Option<f64>) -> str`

执行实例分割

**参数**:
- `confidence_threshold`: 置信度阈值 (0-1, 默认 0.5)
- `iou_threshold`: NMS IoU 阈值 (0-1, 默认 0.3)

**返回**: 检测到的实例数量

### 6.3 HTTP API (instance_seg_server)

#### `POST /api/v1/load_instance_segmentation_model`

**请求**:
```json
{
  "args": {
    "model_path": "/models/pointinst.onnx",
    "model_type": "onnx",
    "device": "cuda"
  }
}
```

**响应**:
```json
{
  "result": {
    "message": "✓ ONNX 模型加载成功",
    "info": {...}
  },
  "error": null
}
```

#### `POST /api/v1/run_instance_segmentation`

**请求**:
```json
{
  "args": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.3
  }
}
```

---

## 7. 技术栈与依赖

### 7.1 Rust 依赖 (`Cargo.toml`)

| 包 | 版本 | 用途 |
|------|------|------|
| `tokitai` | 0.4 | AI 工具定义宏 |
| `tokitai-core` | 0.4 | tokitai 核心类型 |
| `serde` + `serde_derive` | 1.0 | 序列化框架 |
| `serde_json` | 1.0 | JSON 处理 |
| `thiserror` | 1.0 | 错误处理 |
| `reqwest` | 0.11 | HTTP 客户端 |
| `tokio` | 1.0 | 异步运行时 |
| `async-trait` | 0.1 | 异步 trait 支持 |
| `uuid` | 1.0 | UUID 生成 |
| `tracing` + `tracing-subscriber` | 0.1/0.3 | 日志 |

### 7.2 Python 依赖 (`requirements.txt`)

| 包 | 版本 | 用途 |
|------|------|------|
| `numpy` | >=1.24.0 | 数值计算 |
| `onnxruntime` | >=1.15.0 | ONNX CPU 推理 |
| `onnxruntime-gpu` | >=1.15.0 | ONNX GPU 推理 (可选) |
| `open3d` | - | 点云处理 (需系统包) |
| `fastapi` | >=0.100.0 | HTTP 服务 |
| `uvicorn` | >=0.23.0 | ASGI 服务器 |
| `pydantic` | >=2.0.0 | 数据验证 |
| `laspy[laszip]` | - | LAS 格式支持 (可选) |

### 7.3 C++ 依赖

| 依赖 | 用途 |
|------|------|
| `jsoncpp` | JSON 解析 |
| `PCL` | 点云库 (可选) |
| `CUDA` | GPU 加速 (可选) |

### 7.4 系统要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Rust | 1.70 | 1.94+ |
| Python | 3.7 | 3.11-3.12 |
| CMake | 3.10 | 3.20+ |
| CUDA (可选) | 11.0 | 12.0+ |

---

## 8. 开发指南

### 8.1 添加新的 Python 工具

**步骤 1**: 在 `python_tools/pointcloud_tools.py` 实现函数

```python
def your_new_method(param1: str, param2: float) -> dict:
    """
    方法描述

    Args:
        param1: 参数说明
        param2: 参数说明

    Returns:
        dict: {"message": "...", "data": {...}}
    """
    # 实现逻辑
    import open3d as o3d
    # ...

    return {
        "message": "处理完成",
        "result": {...}
    }
```

**步骤 2**: 注册到 `TOOLS` 字典

```python
TOOLS["your_new_method"] = {
    "func": your_new_method,
    "description": "方法的详细描述，AI 会看到这个",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数 1 的详细说明"
            },
            "param2": {
                "type": "number",
                "description": "参数 2 的详细说明"
            }
        },
        "required": ["param1", "param2"]
    }
}
```

**步骤 3**: 在 Rust 端包装 (可选)

```rust
// src/pointcloud_tools.rs
#[tool]
impl PointCloudToolManager {
    pub fn your_new_method(&self, param1: String, param2: f64) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "your_new_method", json!({
            "param1": param1,
            "param2": param2
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }
}
```

### 8.2 添加新的 HTTP 端点

**步骤 1**: 在 `python_tools/instance_seg_server.py` 添加端点

```python
@app.post("/api/v1/your_new_endpoint")
async def your_new_endpoint(request: ToolRequest) -> ToolResponse:
    """新的 HTTP 端点"""
    args = request.args
    
    # 调用工具函数
    result = your_tool_function(**args)
    
    return ToolResponse(result=result, error=None)
```

**步骤 2**: 在 Rust 端通过 HTTP 后端调用

```rust
let switch = BackendSwitch::new_http("http://localhost:8080", None, 30);
let result = switch.call_tool("your_new_endpoint", json!({
    "param1": "value"
}))?;
```

### 8.3 调试技巧

#### 本地测试 Python 工具

```bash
# 点云工具测试
cd python_tools
python pointcloud_tools.py --test

# 实例分割工具测试
python instance_seg_tools.py --test

# PointPillars 测试
python pointpillars_tools.py --test
```

#### 查看 IPC 通信

修改 `src/ipc.rs`，在 `call_tool` 中添加日志:

```rust
tracing::debug!("IPC 发送：{}", request_json);
// ...
tracing::debug!("IPC 接收：{}", response_line);
```

#### 测试 HTTP 服务

```bash
# 启动服务
./python_tools/start_server.sh --port 8080

# 测试端点
curl -X POST http://localhost:8080/api/v1/health
curl -X POST http://localhost:8080/api/v1/load_instance_segmentation_model \
  -H "Content-Type: application/json" \
  -d '{"args": {"model_path": "models/pointinst.onnx"}}'
```

### 8.4 常见问题

#### Q1: Open3D 安装失败

**问题**: Python 3.14 太新，官方 wheel 不支持

**解决**:
```bash
# 方案 A: 使用系统包 (Arch Linux)
sudo pacman -S python-open3d

# 方案 B: 使用 pyenv 安装 Python 3.12
pyenv install 3.12.8
pyenv virtualenv 3.12.8 lidar-env
pyenv local lidar-env
pip install -r requirements.txt
```

#### Q2: 异步运行时冲突

**错误**: `Cannot drop a runtime in a context where blocking is not allowed`

**原因**: `reqwest::blocking` 不能在 tokio 上下文中使用

**解决**: 使用 `std::thread::spawn` + 独立 tokio 运行时 (已在 `backend_switch.rs` 实现)

#### Q3: 点云数据未共享

**问题**: instance_seg_tools.py 找不到点云文件

**解决**: 确保先调用 `pointcloud_tools.py` 的 `load_point_cloud()`,数据会保存到 `tmp/lidar_ai_current_pcd.npy`

---

## 附录

### A. 项目文件结构

```
tokitai-ipc-tool/
├── Cargo.toml                  # Rust 项目配置
├── Cargo.lock                  # 依赖锁定
├── requirements.txt            # Python 依赖
├── SETUP.md                    # 环境配置指南
├── README.md                   # 项目概述
├── .venv/                      # Python 虚拟环境
│
├── src/                        # Rust 源代码
│   ├── lib.rs                  # 库入口
│   ├── main.rs                 # 示例程序
│   ├── error.rs                # 错误类型
│   ├── ipc.rs                  # IPC 通信
│   ├── backend_switch.rs       # 后端切换器
│   ├── ai_scheduler.rs         # AI 调度器
│   ├── pointcloud_tools.rs     # 点云工具管理器
│   ├── instance_seg_tools.rs   # 实例分割工具管理器
│   └── tools/
│       └── mod.rs              # 工具模块
│
├── python_tools/               # Python 工具
│   ├── pointcloud_tools.py     # 点云处理
│   ├── instance_seg_tools.py   # 实例分割
│   ├── pointpillars_tools.py   # PointPillars 检测
│   ├── instance_seg_server.py  # HTTP 服务
│   ├── start_server.sh         # 启动脚本
│   └── models/                 # AI 模型
│       ├── pointpillars.onnx
│       ├── pointpillars_simple.onnx
│       ├── pointpillars_realistic.onnx
│       └── pointpillars_kitti_3class.pth
│
├── cpp_tools/                  # C++ 工具 (框架)
│   ├── pointcloud_tools.cpp
│   └── CMakeLists.txt
│
├── docs/                       # 文档
│   ├── IMPLEMENTATION_STATUS.md
│   ├── NETWORK_SERVICE_DESIGN.md
│   └── ...
│
└── tmp/                        # 临时数据
    └── lidar_ai_current_pcd.npy
```

### B. 快速启动命令

```bash
# 1. 激活 Python 虚拟环境
source .venv/bin/activate

# 2. 构建 Rust 项目
cargo build

# 3. 运行示例
cargo run

# 4. 启动 HTTP 服务 (新终端)
./python_tools/start_server.sh --port 8080

# 5. 启动 Ollama (如果需要 AI 调度)
ollama serve
```

### C. 相关资源

- [tokitai 文档](https://crates.io/crates/tokitai)
- [Open3D 文档](http://www.open3d.org/)
- [Ollama API 参考](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [PointPillars 论文](https://arxiv.org/abs/1812.05784)
