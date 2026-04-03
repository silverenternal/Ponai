# Lidar AI Studio - 快速参考指南

**简明版文档 - 核心概念速查**

---

## 1. 模块总览

```
┌─────────────────────────────────────────────────────────┐
│  模块层次      │  核心组件              │  文件位置    │
├────────────────┼───────────────────────┼──────────────┤
│  AI 调度层      │  AiScheduler          │  src/        │
│  (LLM 工具路由)  │  (Ollama 适配器)       │              │
├────────────────┼───────────────────────┼──────────────┤
│  工具管理层    │  PointCloudToolManager│  src/        │
│  (tokitai 宏)   │  InstanceSegToolManager│             │
│                │  BackendSwitch        │              │
├────────────────┼───────────────────────┼──────────────┤
│  Python 执行层  │  pointcloud_tools.py  │  python_tools/│
│  (点云处理)     │  instance_seg_tools.py│              │
│                │  pointpillars_tools.py│              │
├────────────────┼───────────────────────┼──────────────┤
│  HTTP 服务层    │  instance_seg_server  │  python_tools/│
│  (远程 GPU)     │  FastAPI REST API     │              │
└────────────────┴───────────────────────┴──────────────┘
```

---

## 2. 核心概念

### 2.1 IPC 通信协议

**请求格式** (Rust → Python):
```json
{"tool": "load_point_cloud", "args": {"file_path": "/data.pcd"}}
```

**响应格式** (Python → Rust):
```json
{"result": {"message": "成功"}, "error": null}
```

### 2.2 BackendSwitch 双后端

| 模式 | 创建方法 | 适用场景 |
|------|----------|----------|
| **IPC** | `BackendSwitch::new_ipc(script)` | 本地开发、CPU 推理 |
| **HTTP** | `BackendSwitch::new_http(url, key, timeout)` | 远程 GPU 服务器 |

**动态切换**:
```rust
let mut switch = BackendSwitch::new_ipc("python_tools/instance_seg_tools.py")?;
switch.switch_to_http("http://gpu-server:8080", None, 30);
```

### 2.3 tokitai `#[tool]` 宏

自动将 Rust 方法暴露给 AI 调度器:

```rust
#[tool]
impl PointCloudToolManager {
    /// 加载点云文件（支持 PCD、PLY、LAS 格式）
    pub fn load_point_cloud(&self, file_path: String) -> Result<String, ToolError> {
        // 方法实现
    }
}
```

---

## 3. 工具 API 速查

### 3.1 点云工具 (PointCloudToolManager)

| 方法 | 参数 | 说明 |
|------|------|------|
| `load_point_cloud(file_path)` | file_path: str | 加载 PCD/PLY/LAS |
| `get_point_cloud_info()` | - | 获取点数、边界框 |
| `downsample(voxel_size)` | voxel_size: f64 | 体素降采样 |
| `estimate_normals(k_neighbors)` | k_neighbors: u32 | 法线估计 |
| `remove_outliers(nb_neighbors, std_ratio)` | nb_neighbors: u32, std_ratio: f64 | 离群点移除 |
| `segment_plane(distance_threshold, max_iterations)` | distance_threshold: f64, max_iterations: u32 | RANSAC 平面分割 |
| `euclidean_clustering(tolerance, min_size, max_size)` | tolerance: f64, min/max_size: u32 | DBSCAN 聚类 |
| `save_point_cloud(file_path)` | file_path: str | 保存 PCD/PLY |
| `visualize()` | - | 打开可视化窗口 |

### 3.2 实例分割工具 (InstanceSegToolManager)

| 方法 | 参数 | 说明 |
|------|------|------|
| `load_model(model_path, model_type, device)` | model_path: str, model_type: "onnx"/"pytorch", device: "cpu"/"cuda" | 加载模型 |
| `run_segmentation(conf_thresh, iou_thresh)` | conf_thresh: f64, iou_thresh: f64 | 执行推理 |
| `get_result()` | - | 获取实例列表 |
| `visualize()` | - | 彩色可视化 |
| `export_result(output_path, format)` | output_path: str, format: "json"/"pcd"/"numpy" | 导出结果 |
| `switch_backend(backend_type, config)` | backend_type: "ipc"/"http", config: str | 切换后端 |

### 3.3 PointPillars 工具

| 方法 | 参数 | 说明 |
|------|------|------|
| `load_pointpillars(model_path, device)` | model_path: str, device: "cpu"/"cuda" | 加载 ONNX 模型 |
| `run_pointpillars(points, conf_thresh)` | points: np.ndarray, conf_thresh: f64 | 3D 目标检测 |
| `get_pointpillars_info()` | - | 获取模型信息 |

---

## 4. 数据流向图

### 4.1 AI 工具调用

```
用户输入 → Ollama (LLM) → AiScheduler → PointCloudToolManager
                                              │
                                              ▼
                                         IpcToolRunner
                                              │
                                    (stdin/stdout JSON)
                                              │
                                              ▼
                                     Python: load_point_cloud()
                                              │
                                              ▼
                                     Open3D: 读取 PCD 文件
```

### 4.2 点云数据共享

```
pointcloud_tools.py              instance_seg_tools.py
       │                                │
       ▼                                │
load_point_cloud("/data.pcd")           │
       │                                │
       ▼                                │
np.save("tmp/lidar_ai_current_pcd.npy") │
       │                                │
       └────────────────────────────────┤
                                        ▼
                                np.load("tmp/...npy")
                                        │
                                        ▼
                                run_segmentation()
```

---

## 5. AI 模型清单

### 5.1 本地 LLM

| 模型 | 服务地址 | 用途 |
|------|----------|------|
| llama3.2 | `localhost:11434` | AI 工具调度 |

**启动命令**:
```bash
ollama pull llama3.2
ollama serve
```

### 5.2 点云模型

| 模型 | 位置 | 任务 | 格式 |
|------|------|------|------|
| pointpillars.onnx | python_tools/models/ | 3D 目标检测 | ONNX |
| pointpillars_simple.onnx | python_tools/models/ | 简化版检测 | ONNX |
| pointpillars_realistic.onnx | python_tools/models/ | 真实场景检测 | ONNX |
| pointpillars_kitti_3class.pth | python_tools/models/ | KITTI 检测 | PyTorch |

---

## 6. 快速启动

### 6.1 环境准备

```bash
# 1. 激活 Python 虚拟环境
source .venv/bin/activate

# 2. 验证依赖
python -c "import open3d, numpy, onnxruntime"

# 3. 构建 Rust 项目
cargo build
```

### 6.2 运行示例

```bash
# 运行主程序
cargo run

# 测试 Python 工具
python python_tools/pointcloud_tools.py --test
python python_tools/instance_seg_tools.py --test
python python_tools/pointpillars_tools.py --test
```

### 6.3 启动 HTTP 服务

```bash
# 新终端
./python_tools/start_server.sh --port 8080 --api-key your-key

# 测试端点
curl http://localhost:8080/health
```

---

## 7. 开发者速查

### 7.1 添加 Python 工具

```python
# 1. 实现函数
def my_tool(param1: str, param2: float) -> dict:
    """工具描述"""
    # 实现逻辑
    return {"message": "完成", "data": {...}}

# 2. 注册到 TOOLS 字典
TOOLS["my_tool"] = {
    "func": my_tool,
    "description": "AI 可见的描述",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "说明"}
        },
        "required": ["param1"]
    }
}
```

### 7.2 添加 Rust 工具包装

```rust
#[tool]
impl PointCloudToolManager {
    pub fn my_tool(&self, param1: String) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "my_tool", json!({
            "param1": param1
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }
}
```

### 7.3 添加 HTTP 端点

```python
# instance_seg_server.py
@app.post("/api/v1/my_endpoint")
async def my_endpoint(request: ToolRequest) -> ToolResponse:
    result = my_tool_function(**request.args)
    return ToolResponse(result=result, error=None)
```

---

## 8. 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `open3d 导入失败` | Python 3.14 不支持 | `sudo pacman -S python-open3d` 或用 pyenv |
| `Cannot drop a runtime` | reqwest::blocking 在 tokio 上下文 | 已修复：使用 std::thread::spawn |
| `点云文件不存在` | tmp/lidar_ai_current_pcd.npy 未创建 | 先调用 load_point_cloud() |
| `Ollama 连接失败` | 服务未启动 | `ollama serve` |

---

## 9. 文件路径参考

| 文件 | 路径 |
|------|------|
| Rust 库入口 | `src/lib.rs` |
| 点云工具管理器 | `src/pointcloud_tools.rs` |
| 实例分割工具管理器 | `src/instance_seg_tools.rs` |
| 后端切换器 | `src/backend_switch.rs` |
| AI 调度器 | `src/ai_scheduler.rs` |
| IPC 通信 | `src/ipc.rs` |
| 点云 Python 工具 | `python_tools/pointcloud_tools.py` |
| 实例分割 Python 工具 | `python_tools/instance_seg_tools.py` |
| PointPillars 工具 | `python_tools/pointpillars_tools.py` |
| HTTP 服务 | `python_tools/instance_seg_server.py` |
| 模型文件 | `python_tools/models/*.onnx` |
| 临时数据 | `tmp/lidar_ai_current_pcd.npy` |

---

## 10. 性能参考

| 操作 | IPC 延迟 | HTTP 延迟 (本地) | HTTP 延迟 (远程) |
|------|----------|------------------|------------------|
| load_point_cloud | ~50ms | ~60ms | ~80-150ms |
| run_segmentation | ~100ms | ~110ms | ~150-300ms |
| run_pointpillars | ~50ms | ~60ms | ~100-200ms |

**优化建议**:
- 本地开发：使用 IPC 模式
- 生产部署：HTTP 模式 + GPU 服务器
- 批量处理：使用 `/api/v1/batch_segmentation`

---

## 11. 命令速查表

```bash
# ========== 环境管理 ==========
source .venv/bin/activate           # 激活虚拟环境
cargo build                         # 构建 Rust
cargo run                           # 运行示例

# ========== Python 工具测试 ==========
python python_tools/pointcloud_tools.py --test
python python_tools/instance_seg_tools.py --test
python python_tools/pointpillars_tools.py --test

# ========== HTTP 服务 ==========
./python_tools/start_server.sh --port 8080
curl http://localhost:8080/health

# ========== Ollama ==========
ollama pull llama3.2                # 拉取模型
ollama serve                        # 启动服务
ollama run llama3.2                 # 交互式运行

# ========== C++ 编译 ==========
cd cpp_tools/build && cmake .. && make
```

---

## 12. 关键依赖版本

| 组件 | 包名 | 版本 |
|------|------|------|
| Rust | tokitai | 0.4 |
| Rust | serde_json | 1.0 |
| Rust | reqwest | 0.11 |
| Rust | tokio | 1.0 |
| Python | numpy | >=1.24.0 |
| Python | onnxruntime | >=1.15.0 |
| Python | open3d | 系统包 |
| Python | fastapi | >=0.100.0 |

---

**完整文档**: [`docs/PROJECT_DOCUMENTATION.md`](./docs/PROJECT_DOCUMENTATION.md)
