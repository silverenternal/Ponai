# Lidar AI Studio

**3D 点云车机应用 - AI 调度与跨语言工具框架**

基于 **tokitai** 库构建，通过 IPC/HTTP（进程间通信/网络服务）实现 Rust AI 调度层与 Python/C++ 点云处理工具的无缝集成。

> 📚 **完整文档**: 查看 [`tokitai-ipc-tool/docs/PROJECT_DOCUMENTATION.md`](./tokitai-ipc-tool/docs/PROJECT_DOCUMENTATION.md)
> 
> **最新状态**: 查看 [`tokitai-ipc-tool/docs/IMPLEMENTATION_STATUS.md`](./tokitai-ipc-tool/docs/IMPLEMENTATION_STATUS.md)

## 架构设计

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

## 核心特性

### 🚀 tokitai 优势利用

| tokitai 特性 | 在本架构中的体现 |
|-------------|-----------------|
| **编译时类型安全** | Rust 端 `#[tool]` 宏自动生成类型检查代码 |
| **零运行时依赖** | 可选启用 runtime feature，仅用于 IPC 通信 |
| **厂商中立** | 工具定义可发给任意 LLM（OpenAI、Anthropic、Ollama 等） |
| **单一属性宏** | 只需 `#[tool]`，自动生成 `tool_definitions()` 等方法 |
| **跨语言扩展** | 通过 IPC/HTTP 调用 Python/C++ 等任意语言的工具 |

### 🛠️ 跨语言 IPC/HTTP 双模式

本框架通过 **BackendSwitch** 实现 IPC/HTTP 双后端支持，运行时动态切换：

```
┌─────────────┐      JSON       ┌─────────────┐
│   Rust      │ ◄─────────────► │   Python    │
│  (tokitai)  │   IPC or HTTP   │  (Open3D)   │
│  Switch     │                 │  (ONNX)     │
└─────────────┘                 └─────────────┘
```

**模式对比**:

| 模式 | 延迟 | 适用场景 |
|------|------|----------|
| **IPC** | ~50ms | 本地开发、无 GPU |
| **HTTP** | ~60-150ms | 远程 GPU 服务器、负载均衡 |

**请求格式 (IPC)**:
```json
{"tool": "tool_name", "args": {"param1": "value1", "param2": 42}}
```

**响应格式**:
```json
{"result": {"key": "value"}, "error": null}
```

或
```json
{"result": null, "error": "错误信息"}
```

### 🎯 点云实例分割

支持多种 3D 点云实例分割模型，推荐 **PointPillars**：

| 模型 | 参数量 | 模型大小 | CPU FPS | GPU FPS | 检测类别 |
|------|--------|----------|---------|---------|----------|
| **PointPillars** ⭐ | 5.2M | 15 MB | 18.5 | 95 | Car, Pedestrian, Cyclist |
| SECOND | 8.3M | 24 MB | 14.0 | 58 | Car, Pedestrian, Cyclist |
| PointInst | 12.5M | 35 MB | 15.0 | 72 | Car, Pedestrian, Cyclist |
| CenterPoint | 15.8M | 45 MB | 12.0 | 65 | 6 类 (nuScenes) |

详细对比报告：[`docs/MODEL_COMPARISON_REPORT.md`](./tokitai-ipc-tool/docs/MODEL_COMPARISON_REPORT.md)

## 项目结构

```
lidar-ai-studio/
├── Cargo.toml                  # Rust 项目配置
├── src/
│   ├── lib.rs                  # 库入口
│   ├── main.rs                 # 示例程序
│   ├── error.rs                # 错误类型定义
│   ├── ipc.rs                  # IPC 通信模块
│   ├── ipc_types.rs            # IPC 类型定义
│   ├── ipc_error.rs            # IPC 错误码 (35+ 种)
│   ├── path_utils.rs           # 路径解析工具
│   ├── backend/
│   │   ├── mod.rs              # 后端模块入口
│   │   ├── ipc_backend.rs      # IPC 后端实现
│   │   └── http_backend.rs     # HTTP 后端实现
│   ├── backend_switch.rs       # 后端切换器 (IPC/HTTP)
│   ├── ai_scheduler.rs         # AI 调度层（Ollama 适配）
│   ├── pointcloud_tools.rs     # 点云工具层（tokitai 包装）
│   ├── instance_seg_tools.rs   # 实例分割工具层（双后端支持）
│   └── tools/
│       └── mod.rs              # 工具模块
├── python_tools/
│   ├── pointcloud_tools.py     # Python 点云工具服务 (Open3D)
│   ├── instance_seg_tools.py   # Python 实例分割服务 (ONNX)
│   ├── pointpillars_tools.py   # PointPillars 3D 检测
│   ├── instance_seg_server.py  # HTTP REST API 服务
│   ├── start_server.sh         # HTTP 服务启动脚本
│   ├── models/                 # 预训练模型
│   │   ├── pointpillars.onnx
│   │   ├── pointpillars_simple.onnx
│   │   ├── pointpillars_realistic.onnx
│   │   └── pointpillars_kitti_3class.pth
│   └── requirements.txt        # Python 依赖
├── cpp_tools/
│   ├── pointcloud_tools.cpp    # C++ 点云工具服务（框架）
│   └── CMakeLists.txt          # C++ 构建配置
├── docs/                       # 完整文档
│   ├── PROJECT_DOCUMENTATION.md  # 完整项目文档
│   ├── IMPLEMENTATION_STATUS.md  # 实现状态
│   ├── MODEL_COMPARISON_REPORT.md # 模型对比报告
│   ├── NETWORK_SERVICE_DESIGN.md # 网络服务设计
│   └── INSTANCE_SEG_DESIGN.md    # 实例分割设计
└── README.md                   # 本文档
```

## 快速开始

### 环境要求

- Rust 1.70+ (推荐 1.94+)
- Python 3.11-3.12 (推荐，Open3D 支持最佳)
- （可选）Ollama 服务 - 用于 AI 调度功能
- （可选）CMake + jsoncpp - 用于 C++ 工具

### 构建

```bash
# 构建 Rust 项目
cargo build

# 运行示例
cargo run
```

### 运行输出示例

```
=== 3D 点云车机应用 - AI 调度框架 ===

1. 初始化点云工具层...
   ✓ Python 点云工具已加载

2. 获取工具定义（用于 AI 调用）...
   可用工具：
   - load_point_cloud: 加载点云文件（支持 PCD、PLY、LAS 格式）
   - get_point_cloud_info: 获取点云基本信息（点数、边界框、密度等）
   - downsample: 点云降采样（体素网格滤波）
   - estimate_normals: 法线估计
   - remove_outliers: 离群点移除
   - segment_plane: 平面分割（RANSAC）
   - euclidean_clustering: 欧式聚类
   - save_point_cloud: 保存点云到文件
   - visualize: 可视化点云（打开可视化窗口）
   - get_cpp_tools: 获取可用的 C++ 高性能工具列表

3. 初始化 AI 调度器（Ollama 适配）...
   ✓ AI 调度器已配置（Ollama: localhost:11434）
   ✓ 工具已注册到 AI 调度器

4. 演示点云工具调用...
   加载点云：成功加载点云文件：/data/pointcloud.pcd
   降采样：执行降采样，体素大小：0.05
   ...

5. 实例分割工具演示...
   ✓ 实例分割工具已加载（IPC 模式）
   演示切换到 HTTP 后端...
   ✓ 当前后端：HTTP
   切换回 IPC 后端...
   ✓ 当前后端：IPC

6. AI 调度演示（需要 Ollama 服务）...
   注意：此步骤需要本地运行 Ollama 服务
   启动命令：ollama serve
```

## 点云工具 API

### Python 工具（基于 Open3D 框架）

| 工具 | 描述 | 参数 |
|------|------|------|
| `load_point_cloud` | 加载点云文件 | `file_path: str` |
| `get_point_cloud_info` | 获取点云信息 | - |
| `downsample` | 体素降采样 | `voxel_size: float` |
| `estimate_normals` | 法线估计 | `k_neighbors: int` |
| `remove_outliers` | 离群点移除 | `nb_neighbors: int`, `std_ratio: float` |
| `segment_plane` | 平面分割 (RANSAC) | `distance_threshold: float`, `max_iterations: int` |
| `euclidean_clustering` | 欧式聚类 | `tolerance: float`, `min/max_cluster_size: int` |
| `save_point_cloud` | 保存点云 | `file_path: str` |
| `visualize` | 可视化 | - |

### 实例分割工具（基于 ONNX Runtime）

| 工具 | 描述 | 参数 | 后端 |
|------|------|------|------|
| `load_model` | 加载实例分割模型 | `model_path`, `model_type`, `device` | IPC/HTTP |
| `run_segmentation` | 执行实例分割 | `confidence_threshold`, `iou_threshold` | IPC/HTTP |
| `get_result` | 获取分割结果 | - | IPC/HTTP |
| `visualize` | 可视化结果 | - | IPC/HTTP |
| `export_result` | 导出结果 | `output_path`, `format` | IPC/HTTP |
| `switch_backend` | 动态切换后端 | `backend_type`, `config` | - |

**支持模型**: PointPillars (推荐), PointInst, SECOND, CenterPoint

**输出格式**:
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

### C++ 工具（高性能，待实现）

| 工具 | 描述 |
|------|------|
| `gpu_accelerated_filter` | GPU 加速滤波 |
| `real_time_segmentation` | 实时分割 |
| `cuda_normals` | CUDA 加速法线估计 |

## AI 调度层（Ollama 适配）

### 配置

```rust
use lidar_ai_studio::{AiScheduler, AiSchedulerConfig};

let config = AiSchedulerConfig {
    host: "localhost".to_string(),
    port: 11434,
    model: "llama3.2".to_string(),
    stream: false,
};

let scheduler = AiScheduler::new(config);
```

### 使用示例

```rust
// 注册工具到 AI 调度器
scheduler.register_tools(tool_definitions).await;

// 发送消息给 AI
let response = scheduler.chat("帮我加载点云并进行降采样").await?;

// 处理 AI 的工具调用请求
if let Some(tool_calls) = response.tool_calls {
    for tool_call in tool_calls {
        // 执行工具调用
        let result = tools.call_tool(&tool_call.function.name, ...)?;

        // 将结果返回给 AI
        let final_response = scheduler
            .process_tool_result(&tool_call, result)
            .await?;
    }
}
```

### 启动 Ollama 服务

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull llama3.2

# 启动服务
ollama serve
```

## BackendSwitch: IPC/HTTP 双后端

### 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   InstanceSegToolManager                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              BackendSwitch (切换器)                        │  │
│  │  ┌─────────────┐  ┌─────────────┐                         │  │
│  │  │ IPC Backend │  │ HTTP Backend│                         │  │
│  │  │  (本地)     │  │  (网络)     │                         │  │
│  │  └─────────────┘  └─────────────┘                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 使用示例

```rust
use lidar_ai_studio::InstanceSegToolManager;

// 创建 IPC 后端（本地开发）
let seg_tool = InstanceSegToolManager::new_ipc("python_tools/instance_seg_tools.py")?;

// 创建 HTTP 后端（远程 GPU 服务器）
let seg_tool = InstanceSegToolManager::new_http(
    "http://gpu-server:8080",
    Some("your-api-key".to_string()),
    60,
);

// 动态切换后端
seg_tool.switch_backend("http".to_string(), Some("http://gpu-server:8080".to_string()))?;
seg_tool.switch_backend("ipc".to_string(), None)?;
```

### 模式对比

| 模式 | 延迟 | 吞吐量 | 适用场景 |
|------|------|--------|----------|
| **IPC** | ~50ms | 单进程 | 本地开发、无 GPU |
| **HTTP (本地)** | ~60ms | 可多 worker | 本地 GPU |
| **HTTP (远程)** | ~80-150ms | 高 | 远程 GPU 服务器 |
| **HTTP (批处理)** | ~200ms | 很高 | 批量推理 |

## 开发者指南：添加工具方法

### Python 开发者

**文件位置：** `python_tools/pointcloud_tools.py` 或 `python_tools/instance_seg_tools.py`

**步骤 1: 实现点云处理函数**

```python
def your_pointcloud_method(param1: str, param2: float) -> dict:
    """
    你的方法描述

    Args:
        param1: 参数说明
        param2: 参数说明

    Returns:
        dict: 返回结果（必须是 JSON 可序列化的字典）
    """
    # 在这里实现你的点云算法
    # 可以使用 Open3D、NumPy、PCL 等库

    import open3d as o3d
    import numpy as np

    # 示例：加载点云并处理
    # pcd = o3d.io.read_point_cloud(param1)
    # ... 你的处理逻辑 ...

    return {
        "message": "处理完成",
        "result": {...}  # 可选的结果数据
    }
```

**步骤 2: 注册到 TOOLS 字典**

```python
# 在文件的 TOOLS 字典中添加你的工具
TOOLS["your_method_name"] = {
    "func": your_pointcloud_method,  # 函数引用
    "description": "方法的详细描述，AI 会看到这个描述来决定是否调用",
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
        "required": ["param1", "param2"]  # 必填参数列表
    }
}
```

**完整示例位置：**
- 打开 `python_tools/pointcloud_tools.py`
- 查看现有的工具实现（如 `load_point_cloud`, `downsample` 等）
- 参考 `TOOLS` 字典的注册格式

---

### C++ 开发者

**文件位置：** `cpp_tools/pointcloud_tools.cpp`

**步骤 1: 实现点云处理函数**

```cpp
#include <json/json.h>
// #include <pcl/...>  // 使用 PCL 库
// #include <open3d/...>  // 或使用 Open3D

namespace lidar_tools {

Json::Value your_pointcloud_method(const Json::Value& args) {
    Json::Value result;

    // 从 args 获取参数
    std::string param1 = args["param1"].asString();
    double param2 = args.get("param2", 0.0).asDouble();

    // 在这里实现你的点云算法
    // 可以使用 PCL、Open3D、CUDA 等库

    // 示例：
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::io::loadPCDFile(param1, *cloud);
    // ... 你的处理逻辑 ...

    result["message"] = "处理完成";
    result["output"] = "...";  // 可选的输出数据

    return result;
}

}  // namespace lidar_tools
```

**步骤 2: 注册到工具映射表**

```cpp
// 在 g_tools 映射表中注册你的工具
std::map<std::string, ToolFunction> g_tools = {
    {"your_method_name", your_pointcloud_method},  // 添加这一行
    // ... 其他工具
};
```

**步骤 3: 编译 C++ 工具**

```bash
# 创建构建目录
mkdir -p cpp_tools/build && cd cpp_tools/build

# 配置 CMake
cmake ..

# 编译
make

# 生成的二进制文件：cpp_tools/build/pointcloud_tools_cpp
```

**完整示例位置：**
- 打开 `cpp_tools/pointcloud_tools.cpp`
- 查看现有的工具实现（如 `load_point_cloud`, `gpu_accelerated_filter` 等）
- 参考 `g_tools` 映射表的注册格式

---

### Rust 开发者：将新工具暴露给 AI

**文件位置：** `src/pointcloud_tools.rs` 或 `src/instance_seg_tools.rs`

在 `#[tool]` impl 块中添加新方法：

```rust
#[tool]
impl PointCloudToolManager {
    /// 方法的详细描述（AI 会看到这个）
    pub fn your_method_name(&self, param1: String, param2: f64) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "your_method_name", json!({
            "param1": param1,
            "param2": param2
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;

        Ok(result["message"].as_str().unwrap_or("").to_string())
    }
}
```

**参数说明：**
- 第一个参数 `"python"` 或 `"cpp"`：选择后端
- 第二个参数：与 Python/C++ 中的函数名一致
- 第三个参数：JSON 格式的参数传递

---

### HTTP 服务开发者

**文件位置：** `python_tools/instance_seg_server.py`

**添加新的 HTTP 端点：**

```python
from fastapi import APIRouter

router = APIRouter()

@app.post("/api/v1/your_new_endpoint")
async def your_new_endpoint(request: ToolRequest) -> ToolResponse:
    """新的 HTTP 端点"""
    args = request.args
    
    # 调用工具函数
    result = your_tool_function(**args)
    
    return ToolResponse(result=result, error=None)
```

**完整示例位置：**
- 打开 `python_tools/instance_seg_server.py`
- 查看现有的端点实现
- 参考 FastAPI 路由格式

---

## 依赖说明

### Rust 依赖 (`Cargo.toml`)

```toml
[dependencies]
tokitai = "0.4"           # AI 工具定义宏
tokitai-core = "0.4"      # 核心类型
serde = { version = "1.0", features = ["derive"] }  # 序列化
serde_json = "1.0"        # JSON 处理
thiserror = "1.0"         # 错误处理
reqwest = { version = "0.12", features = ["json"] } # HTTP 客户端
tokio = { version = "1.35", features = ["full"] }   # 异步运行时
async-trait = "0.1"       # 异步 trait
uuid = { version = "1.0", features = ["v4"] }       # UUID
tracing = "0.1"           # 日志
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
once_cell = "1.19"        # 全局静态
```

### Python 依赖 (`requirements.txt`)

```
numpy>=1.24.0             # 数值计算
onnxruntime>=1.15.0       # ONNX CPU 推理
# onnxruntime-gpu         # ONNX GPU 推理 (可选)
open3d                    # 点云处理 (需系统包)
fastapi>=0.100.0          # HTTP 服务
uvicorn[standard]>=0.23.0 # ASGI 服务器
pydantic>=2.0.0           # 数据验证
```

### C++ 依赖

| 依赖 | 用途 |
|------|------|
| `jsoncpp` | JSON 解析 |
| `PCL` (可选) | 点云库 |
| `CUDA` (可选) | GPU 加速 |

### 系统要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Rust | 1.70 | 1.94+ |
| Python | 3.7 | 3.11-3.12 |
| CMake (可选) | 3.10 | 3.20+ |
| CUDA (可选) | 11.0 | 12.0+ |

## 许可证

MIT OR Apache-2.0

## 致谢

- [tokitai](https://crates.io/crates/tokitai) - AI 工具定义框架
- [Open3D](http://www.open3d.org/) - 3D 数据处理库
- [PCL](https://pointclouds.org/) - 点云库
- [Ollama](https://ollama.com/) - 本地 LLM 服务
- [MMDetection3D](https://mmdetection3d.readthedocs.io/) - 3D 目标检测框架
- [ONNX Runtime](https://onnxruntime.ai/) - 模型推理引擎

## 相关文档

- [完整项目文档](./tokitai-ipc-tool/docs/PROJECT_DOCUMENTATION.md) - 详细的架构和 API 参考
- [实现状态](./tokitai-ipc-tool/docs/IMPLEMENTATION_STATUS.md) - 当前实现进度
- [模型对比报告](./tokitai-ipc-tool/docs/MODEL_COMPARISON_REPORT.md) - PointPillars 等模型性能对比
- [环境配置](./tokitai-ipc-tool/SETUP.md) - 开发和部署指南
