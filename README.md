# Lidar AI Studio

**3D 点云车机应用 - AI 调度与跨语言工具框架**

基于 **tokitai** 库构建，通过 IPC（进程间通信）实现 Rust AI 调度层与 Python/C++ 点云处理工具的无缝集成。

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
| **跨语言扩展** | 通过 IPC 调用 Python/C++ 等任意语言的工具 |

### 🛠️ 跨语言 IPC

本框架通过 **stdin/stdout JSON Lines** 实现跨语言工具调用：

```
┌─────────────┐      JSON       ┌─────────────┐
│   Rust      │ ◄─────────────► │   Python    │
│  (tokitai)  │   stdin/stdout  │  (Open3D)   │
└─────────────┘                 └─────────────┘
```

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

## 项目结构

```
lidar-ai-studio/
├── Cargo.toml                  # Rust 项目配置
├── src/
│   ├── lib.rs                  # 库入口
│   ├── main.rs                 # 示例程序
│   ├── error.rs                # 错误类型定义
│   ├── ipc.rs                  # IPC 通信模块
│   ├── ai_scheduler.rs         # AI 调度层（Ollama 适配）
│   ├── pointcloud_tools.rs     # 点云工具层（tokitai 包装）
│   └── tools/
│       └── mod.rs              # 工具模块
├── python_tools/
│   └── pointcloud_tools.py     # Python 点云工具服务
├── cpp_tools/
│   ├── pointcloud_tools.cpp    # C++ 点云工具服务（框架）
│   └── CMakeLists.txt          # C++ 构建配置
└── README.md                   # 本文档
```

## 快速开始

### 环境要求

- Rust 1.70+
- Python 3.7+
- （可选）Ollama 服务 - 用于 AI 调度功能

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

## 开发者指南：添加点云处理方法

### Python 开发者

**文件位置：** `python_tools/pointcloud_tools.py`

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
        "result": "处理结果",
        "output_data": [...]  # 可选的输出数据
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

**文件位置：** `src/pointcloud_tools.rs`

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
        
        Ok(result["result"].as_str().unwrap_or("").to_string())
    }
}
```

**参数说明：**
- 第一个参数 `"python"` 或 `"cpp"`：选择后端
- 第二个参数：与 Python/C++ 中的函数名一致
- 第三个参数：JSON 格式的参数传递

---

## 依赖说明

```toml
[dependencies]
tokitai = "0.4"           # AI 工具定义宏
tokitai-core = "0.4"      # 核心类型
reqwest = "0.11"          # HTTP 客户端（Ollama API）
tokio = "1.0"             # 异步运行时
serde_json = "1.0"        # JSON 处理
```

## 许可证

MIT OR Apache-2.0

## 致谢

- [tokitai](https://crates.io/crates/tokitai) - AI 工具定义框架
- [Open3D](http://www.open3d.org/) - 3D 数据处理库
- [PCL](https://pointclouds.org/) - 点云库
- [Ollama](https://ollama.com/) - 本地 LLM 服务
