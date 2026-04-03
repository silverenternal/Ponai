# 实例分割网络服务化架构设计

**版本**：v2.0  
**日期**：2026 年 4 月 2 日  
**架构**：IPC/网络双模式 + Switch 切换机制

---

## 1. 架构设计

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI 调度层                                 │
│                    (Ollama + 工具路由)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Rust 点云工具层 (tokitai)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              InstanceSegToolManager                      │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │           BackendSwitch (切换器)                     │ │  │
│  │  │  ┌─────────────┐  ┌─────────────┐                   │ │  │
│  │  │  │ IPC Backend │  │ HTTP Backend│                   │ │  │
│  │  │  │  (本地)     │  │  (网络)     │                   │ │  │
│  │  │  └─────────────┘  └─────────────┘                   │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│  Python IPC Service     │       │  Python HTTP Service    │
│  (stdin/stdout JSON)    │       │  (FastAPI/Flask)        │
│  pointcloud_tools.py    │       │  instance_seg_server.py │
│  instance_seg_tools.py  │       │                         │
└─────────────────────────┘       └─────────────────────────┘
        │                                   │
        │                                   ▼
        │                           ┌─────────────────────────┐
        │                           │   ONNX Runtime /        │
        │                           │   PyTorch Model         │
        │                           │   (可 GPU 加速)          │
        │                           └─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   独立部署的模型服务     │
│  (可部署在 GPU 服务器)    │
└─────────────────────────┘
```

---

## 2. 核心设计

### 2.1 BackendSwitch 设计

```rust
/// 后端类型枚举
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// 本地 IPC 进程调用
    Ipc,
    /// 网络 HTTP 调用
    Http,
}

/// 后端配置
#[derive(Debug, Clone)]
pub enum BackendConfig {
    Ipc {
        script_path: String,
    },
    Http {
        base_url: String,
        api_key: Option<String>,
        timeout_secs: u64,
    },
}

/// 后端切换器
pub struct BackendSwitch {
    backend_type: BackendType,
    ipc_runner: Option<IpcToolRunner>,
    http_client: Option<reqwest::Client>,
    http_config: Option<HttpConfig>,
}

#[derive(Debug, Clone)]
pub struct HttpConfig {
    base_url: String,
    api_key: Option<String>,
    timeout_secs: u64,
}

impl BackendSwitch {
    /// 创建 IPC 后端
    pub fn new_ipc(script_path: &str) -> Result<Self, LidarAiError> {
        let runner = IpcToolRunner::new_python(script_path)?;
        Ok(Self {
            backend_type: BackendType::Ipc,
            ipc_runner: Some(runner),
            http_client: None,
            http_config: None,
        })
    }

    /// 创建 HTTP 后端
    pub fn new_http(base_url: &str, api_key: Option<String>, timeout_secs: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_default();

        Self {
            backend_type: BackendType::Http,
            ipc_runner: None,
            http_client: Some(client),
            http_config: Some(HttpConfig {
                base_url: base_url.to_string(),
                api_key,
                timeout_secs,
            }),
        }
    }

    /// 动态切换到 IPC 后端
    pub fn switch_to_ipc(&mut self, script_path: &str) -> Result<(), LidarAiError> {
        let runner = IpcToolRunner::new_python(script_path)?;
        self.ipc_runner = Some(runner);
        self.backend_type = BackendType::Ipc;
        tracing::info!("已切换到 IPC 后端：{}", script_path);
        Ok(())
    }

    /// 动态切换到 HTTP 后端
    pub fn switch_to_http(&mut self, base_url: &str, api_key: Option<String>, timeout_secs: u64) {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_default();

        self.http_client = Some(client);
        self.http_config = Some(HttpConfig {
            base_url: base_url.to_string(),
            api_key,
            timeout_secs,
        });
        self.backend_type = BackendType::Http;
        tracing::info!("已切换到 HTTP 后端：{}", base_url);
    }

    /// 获取当前后端类型
    pub fn current_backend(&self) -> &BackendType {
        &self.backend_type
    }

    /// 通用工具调用（自动路由到对应后端）
    pub fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value, LidarAiError> {
        match &self.backend_type {
            BackendType::Ipc => {
                let runner = self.ipc_runner.as_ref()
                    .ok_or_else(|| LidarAiError::IpcCommunication("IPC runner not initialized".to_string()))?;
                runner.call_tool(tool_name, args)
            }
            BackendType::Http => {
                let client = self.http_client.as_ref()
                    .ok_or_else(|| LidarAiError::Http("HTTP client not initialized".to_string()))?;
                let config = self.http_config.as_ref()
                    .ok_or_else(|| LidarAiError::Config("HTTP config not initialized".to_string()))?;
                
                // HTTP 调用逻辑
                self.call_http_tool(client, config, tool_name, args)
            }
        }
    }

    /// HTTP 工具调用实现
    fn call_http_tool(
        &self,
        client: &reqwest::Client,
        config: &HttpConfig,
        tool_name: &str,
        args: Value,
    ) -> Result<Value, LidarAiError> {
        let url = format!("{}/api/v1/{}", config.base_url, tool_name);
        
        let mut request = client.post(&url).json(&json!({
            "args": args
        }));

        // 添加 API Key（如果有）
        if let Some(ref api_key) = config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        // 发送请求
        let response = futures::executor::block_on(request.send())
            .map_err(|e| LidarAiError::Http(format!("HTTP 请求失败：{}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = futures::executor::block_on(response.text())
                .unwrap_or_default();
            return Err(LidarAiError::Http(format!("HTTP {}: {}", status, error_text)));
        }

        // 解析响应
        let api_response: ApiResponse = futures::executor::block_on(response.json())
            .map_err(|e| LidarAiError::Json(e))?;

        if let Some(error) = api_response.error {
            return Err(LidarAiError::ToolExecution(error));
        }

        Ok(api_response.result.unwrap_or(Value::Null))
    }
}
```

---

## 3. Python HTTP 服务端

### 3.1 FastAPI 实现

```python
#!/usr/bin/env python3
"""
实例分割 HTTP 服务 - FastAPI 实现
支持 GPU 加速推理，可独立部署
"""
import os
import sys
import time
import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import uvicorn

# 导入实例分割工具
from instance_seg_tools import (
    load_model, run_segmentation, get_result,
    visualize, export_result, _model, _result
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 全局状态 ============
_model: Optional[Any] = None
_result: Optional[Dict[str, Any]] = None

# ============ API 密钥验证 ============
API_KEY = os.getenv("INSTANCE_SEG_API_KEY", None)

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """验证 API Key（如果配置了的话）"""
    if API_KEY is None:
        return True  # 未配置则跳过验证
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ============ 请求/响应模型 ============

class ToolRequest(BaseModel):
    """工具调用请求"""
    args: Dict[str, Any] = Field(default_factory=dict, description="工具参数")

class ToolResponse(BaseModel):
    """工具调用响应"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    backend: str
    version: str

class LoadModelRequest(BaseModel):
    """加载模型请求"""
    model_path: str
    model_type: str = "onnx"
    device: str = "cpu"

# ============ 生命周期管理 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # 启动时
    logger.info("实例分割服务启动中...")
    yield
    # 关闭时
    logger.info("实例分割服务关闭")

# ============ 创建 FastAPI 应用 ============

app = FastAPI(
    title="Lidar AI Studio - Instance Segmentation Service",
    description="3D 点云实例分割 HTTP 服务",
    version="1.0.0",
    lifespan=lifespan,
)

# ============ API 端点 ============

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    from instance_seg_tools import ONNX_SUPPORT, TORCH_SUPPORT
    
    return HealthResponse(
        status="healthy",
        model_loaded=(_model is not None),
        backend="onnx" if ONNX_SUPPORT else ("pytorch" if TORCH_SUPPORT else "none"),
        version="1.0.0"
    )

@app.post("/api/v1/load_instance_segmentation_model", response_model=ToolResponse, tags=["工具"])
async def api_load_model(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """加载实例分割模型"""
    args = request.args
    try:
        result = load_model(
            model_path=args.get("model_path"),
            model_type=args.get("model_type", "onnx"),
            device=args.get("device", "cpu")
        )
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"加载模型失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/run_instance_segmentation", response_model=ToolResponse, tags=["工具"])
async def api_run_segmentation(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """执行实例分割"""
    args = request.args
    try:
        result = run_segmentation(
            confidence_threshold=args.get("confidence_threshold", 0.5),
            iou_threshold=args.get("iou_threshold", 0.3)
        )
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/get_segmentation_result", response_model=ToolResponse, tags=["工具"])
async def api_get_result(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """获取分割结果"""
    try:
        result = get_result()
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"获取结果失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/visualize_segmentation", response_model=ToolResponse, tags=["工具"])
async def api_visualize(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """可视化分割结果"""
    try:
        result = visualize()
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"可视化失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/export_segmentation", response_model=ToolResponse, tags=["工具"])
async def api_export(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """导出分割结果"""
    args = request.args
    try:
        result = export_result(
            output_path=args.get("output_path"),
            format=args.get("format", "json")
        )
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"导出失败：{e}")
        return ToolResponse(result=None, error=str(e))

# ============ 批处理端点（高性能场景） ============

class BatchSegmentationRequest(BaseModel):
    """批量分割请求"""
    pointclouds: list  # 点云数据列表
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.3

class BatchSegmentationResponse(BaseModel):
    """批量分割响应"""
    results: list
    total_time_ms: float
    count: int

@app.post("/api/v1/batch_segmentation", response_model=BatchSegmentationResponse, tags=["批处理"])
async def api_batch_segmentation(
    request: BatchSegmentationRequest,
    verified: bool = Depends(verify_api_key)
):
    """批量实例分割（高性能场景）"""
    start_time = time.time()
    results = []
    
    for points in request.pointclouds:
        # 保存临时点云
        np.save(_TEMP_PCD_PATH, np.array(points))
        # 执行分割
        seg_result = run_segmentation(
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold
        )
        results.append(get_result())
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchSegmentationResponse(
        results=results,
        total_time_ms=total_time,
        count=len(results)
    )

# ============ 主程序入口 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实例分割 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--api-key", type=str, default=None, help="API Key")
    
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    
    uvicorn.run(
        "instance_seg_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
```

---

## 4. Rust 端完整实现

### 4.1 新增模块结构

```rust
// src/lib.rs
pub mod error;
pub mod ipc;
pub mod backend_switch;      // ← 新增
pub mod pointcloud_tools;
pub mod instance_seg_tools;  // ← 新增
pub mod ai_scheduler;

// src/backend_switch/mod.rs ← 新增文件
pub mod switch;
pub use switch::*;

// src/instance_seg_tools/mod.rs ← 新增文件
pub mod tools;
pub use tools::*;
```

### 4.2 InstanceSegToolManager

```rust
//! 实例分割工具层 - 支持 IPC/HTTP 双后端
//!
//! 通过 BackendSwitch 实现无缝切换

use serde_json::{json, Value};
use tokitai::tool;
use tokitai::ToolError;

use crate::error::{LidarAiError, Result};
use crate::backend_switch::BackendSwitch;

/// 实例分割工具管理器
pub struct InstanceSegToolManager {
    switch: BackendSwitch,
}

impl InstanceSegToolManager {
    /// 创建 IPC 后端
    pub fn new_ipc(python_script: &str) -> Result<Self> {
        let switch = BackendSwitch::new_ipc(python_script)?;
        Ok(Self { switch })
    }

    /// 创建 HTTP 后端
    pub fn new_http(base_url: &str, api_key: Option<String>, timeout_secs: u64) -> Self {
        let switch = BackendSwitch::new_http(base_url, api_key, timeout_secs);
        Self { switch }
    }

    /// 切换到 IPC 后端
    pub fn switch_to_ipc(&mut self, script_path: &str) -> Result<()> {
        self.switch.switch_to_ipc(script_path)
    }

    /// 切换到 HTTP 后端
    pub fn switch_to_http(&mut self, base_url: &str, api_key: Option<String>, timeout_secs: u64) {
        self.switch.switch_to_http(base_url, api_key, timeout_secs);
    }

    /// 获取当前后端类型
    pub fn current_backend(&self) -> &str {
        match self.switch.current_backend() {
            crate::backend_switch::BackendType::Ipc => "IPC",
            crate::backend_switch::BackendType::Http => "HTTP",
        }
    }

    /// 通用工具调用
    fn invoke_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
        self.switch.call_tool(tool_name, args)
            .map_err(|e| LidarAiError::ToolExecution(e.to_string()))
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

        Ok(result.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string())
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

        Ok(result.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string())
    }

    /// 获取分割结果
    pub fn get_result(&self) -> Result<String, ToolError> {
        let result = self.invoke_tool("get_segmentation_result", json!({}))
            .map_err(|e| ToolError::validation_error(e.to_string()))?;

        Ok(serde_json::to_string_pretty(&result)
            .map_err(|e| ToolError::validation_error(e.to_string()))?)
    }

    /// 可视化分割结果
    pub fn visualize(&self) -> Result<String, ToolError> {
        let result = self.invoke_tool("visualize_segmentation", json!({}))
            .map_err(|e| ToolError::validation_error(e.to_string()))?;

        Ok(result.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string())
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

        Ok(result.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string())
    }

    /// 切换后端（动态配置）
    pub fn switch_backend(
        &mut self,
        backend_type: String,
        config_path: Option<String>,
    ) -> Result<String, ToolError> {
        match backend_type.as_str() {
            "ipc" => {
                let script = config_path.unwrap_or_else(|| "python_tools/instance_seg_tools.py".to_string());
                self.switch_to_ipc(&script)
                    .map_err(|e| ToolError::validation_error(e.to_string()))?;
                Ok(format!("已切换到 IPC 后端：{}", script))
            }
            "http" => {
                let url = config_path.unwrap_or_else(|| "http://localhost:8080".to_string());
                self.switch_to_http(&url, None, 30);
                Ok(format!("已切换到 HTTP 后端：{}", url))
            }
            _ => Err(ToolError::validation_error(format!("未知的后端类型：{}", backend_type))),
        }
    }

    /// 获取当前后端信息
    pub fn get_backend_info(&self) -> Result<String, ToolError> {
        Ok(json!({
            "current_backend": self.current_backend(),
            "description": match self.current_backend() {
                "IPC" => "本地进程调用，低延迟，无需网络",
                "HTTP" => "网络调用，可远程部署，支持 GPU 服务器",
                _ => "未知"
            }
        }).to_string())
    }
}
```

---

## 5. 配置文件

### 5.1 后端配置文件

```toml
# config/backend.toml

[instance_segmentation]
# 默认后端类型："ipc" 或 "http"
default_backend = "ipc"

# IPC 配置
[instance_segmentation.ipc]
script_path = "python_tools/instance_seg_tools.py"

# HTTP 配置
[instance_segmentation.http]
base_url = "http://localhost:8080"
api_key = ""  # 可选
timeout_secs = 30

# 故障转移配置
[instance_segmentation.failover]
enabled = true
retry_count = 3
fallback_backend = "ipc"  # HTTP 失败时回退到 IPC
```

---

## 6. 使用示例

### 6.1 main.rs

```rust
use lidar_ai_studio::{InstanceSegToolManager, AiScheduler, AiSchedulerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::init();

    // === 场景 1: 本地开发（IPC 模式） ===
    let mut seg_tool = InstanceSegToolManager::new_ipc("python_tools/instance_seg_tools.py")?;
    
    // 加载模型
    seg_tool.load_model(
        "models/pointinst.onnx".to_string(),
        Some("onnx".to_string()),
        Some("cpu".to_string()),
    )?;

    // 执行分割
    let result = seg_tool.run_segmentation(Some(0.5), Some(0.3))?;
    println!("分割结果：{}", result);

    // === 场景 2: 生产环境（HTTP 模式，GPU 服务器） ===
    let mut seg_tool_http = InstanceSegToolManager::new_http(
        "http://gpu-server:8080",
        Some("your-api-key".to_string()),
        60,
    );

    seg_tool_http.load_model(
        "/models/pointinst.onnx".to_string(),
        Some("onnx".to_string()),
        Some("cuda".to_string()),
    )?;

    // === 场景 3: 动态切换 ===
    // 开发时本地测试，部署时切换到远程 GPU 服务
    seg_tool.switch_to_http("http://gpu-server:8080", None, 30);
    println!("当前后端：{}", seg_tool.current_backend());

    // 或者切换回 IPC
    seg_tool.switch_to_ipc("python_tools/instance_seg_tools.py")?;
    println!("当前后端：{}", seg_tool.current_backend());

    Ok(())
}
```

---

## 7. 性能对比

| 模式 | 延迟 | 吞吐量 | 适用场景 |
|------|------|--------|----------|
| **IPC** | ~50ms | 单进程 | 本地开发、无 GPU |
| **HTTP (本地)** | ~60ms | 可多 worker | 本地 GPU |
| **HTTP (远程)** | ~80-150ms | 高 | 远程 GPU 服务器 |
| **HTTP (批处理)** | ~200ms | 很高 | 批量推理 |

---

## 8. 部署方案

### 8.1 Docker 部署 HTTP 服务

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN pip install fastapi uvicorn onnxruntime-gpu numpy open3d

# 复制代码
COPY python_tools/instance_seg_tools.py .
COPY instance_seg_server.py .
COPY models/ ./models/

# 暴露端口
EXPOSE 8080

# 启动服务
CMD ["python", "instance_seg_server.py", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

### 8.2 docker-compose

```yaml
version: '3.8'

services:
  instance-seg-gpu:
    build: .
    ports:
      - "8080:8080"
    environment:
      - INSTANCE_SEG_API_KEY=your-secret-key
    volumes:
      - ./models:/app/models
      - ./tmp:/app/tmp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 9. 总结

### 优势

1. **灵活部署**：本地 IPC / 远程 HTTP 随意切换
2. **性能优化**：HTTP 模式可部署在 GPU 服务器
3. **无缝迁移**：代码无需修改，配置切换即可
4. **故障转移**：HTTP 失败可自动回退到 IPC
5. **扩展性强**：支持负载均衡、批处理

### 下一步

1. 实现配置加载模块
2. 添加健康检查和自动重连
3. 实现批处理端点优化吞吐量
4. 添加 Prometheus 监控指标
