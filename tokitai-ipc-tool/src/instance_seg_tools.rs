//! 实例分割工具层 - 支持 IPC/HTTP 双后端
//!
//! 通过 BackendSwitch 实现无缝切换本地进程调用和网络调用

use serde_json::{json, Value};
use tokitai::tool;
use tokitai::ToolError;

use crate::backend_switch::BackendSwitch;
use crate::backend::BackendType;
use crate::error::LidarAiError;

/// 实例分割工具管理器
pub struct InstanceSegToolManager {
    switch: BackendSwitch,
}

impl InstanceSegToolManager {
    /// 创建 IPC 后端
    pub fn new_ipc(python_script: &str) -> std::result::Result<Self, LidarAiError> {
        let switch = BackendSwitch::new_ipc(python_script)?;
        Ok(Self { switch })
    }

    /// 创建 HTTP 后端
    pub fn new_http(base_url: &str, api_key: Option<String>, timeout_secs: u64) -> Self {
        let switch = BackendSwitch::new_http(base_url, api_key, timeout_secs);
        Self { switch }
    }

    /// 切换到 IPC 后端
    pub fn switch_to_ipc(&mut self, script_path: &str) -> std::result::Result<(), LidarAiError> {
        self.switch.switch_to_ipc(script_path)
    }

    /// 切换到 HTTP 后端
    pub fn switch_to_http(&mut self, base_url: &str, api_key: Option<String>, timeout_secs: u64) -> std::result::Result<(), LidarAiError> {
        self.switch.switch_to_http(base_url, api_key, timeout_secs)
    }

    /// 获取当前后端类型
    pub fn current_backend(&self) -> &str {
        match self.switch.current_backend() {
            BackendType::Ipc => "IPC",
            BackendType::Http => "HTTP",
        }
    }

    /// 通用工具调用方法
    fn invoke_tool(&self, tool_name: &str, args: Value) -> std::result::Result<Value, LidarAiError> {
        self.switch.call_tool(tool_name, args)
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
    ) -> std::result::Result<String, ToolError> {
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
    ) -> std::result::Result<String, ToolError> {
        let result = self.invoke_tool("run_instance_segmentation", json!({
            "confidence_threshold": confidence_threshold.unwrap_or(0.5),
            "iou_threshold": iou_threshold.unwrap_or(0.3)
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;

        Ok(result.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string())
    }

    /// 获取分割结果
    pub fn get_result(&self) -> std::result::Result<String, ToolError> {
        let result = self.invoke_tool("get_segmentation_result", json!({}))
            .map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(serde_json::to_string_pretty(&result).map_err(|e| ToolError::validation_error(e.to_string()))?)
    }

    /// 可视化分割结果
    pub fn visualize(&self) -> std::result::Result<String, ToolError> {
        let result = self.invoke_tool("visualize_segmentation", json!({}))
            .map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string())
    }

    /// 导出分割结果
    pub fn export_result(
        &self,
        output_path: String,
        format: Option<String>,
    ) -> std::result::Result<String, ToolError> {
        let result = self.invoke_tool("export_segmentation", json!({
            "output_path": output_path,
            "format": format.unwrap_or("json".to_string())
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;

        Ok(result.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string())
    }

    /// 切换后端（动态配置）
    pub fn switch_backend(
        &self,
        backend_type: String,
        config: Option<String>,
    ) -> std::result::Result<String, ToolError> {
        match backend_type.as_str() {
            "ipc" => {
                let script = config.unwrap_or_else(|| "python_tools/instance_seg_tools.py".to_string());
                // 需要获取 mutable reference
                let mut switch = self.switch.clone();
                switch.switch_to_ipc(&script)
                    .map_err(|e| ToolError::validation_error(e.to_string()))?;
                Ok(format!("✓ 已切换到 IPC 后端：{}", script))
            }
            "http" => {
                let url = config.unwrap_or_else(|| "http://localhost:8080".to_string());
                let mut switch = self.switch.clone();
                switch.switch_to_http(&url, None, 30)
                    .map_err(|e| ToolError::validation_error(e.to_string()))?;
                Ok(format!("✓ 已切换到 HTTP 后端：{}", url))
            }
            _ => Err(ToolError::validation_error(format!("未知的后端类型：{}", backend_type))),
        }
    }

    /// 获取当前后端信息
    pub fn get_backend_info(&self) -> std::result::Result<String, ToolError> {
        let backend = self.current_backend();
        let description = match backend {
            "IPC" => "本地进程调用 - 低延迟，无需网络，适合开发和无 GPU 环境",
            "HTTP" => "网络调用 - 可远程部署到 GPU 服务器，支持负载均衡和高并发",
            _ => "未知后端",
        };

        Ok(json!({
            "current_backend": backend,
            "description": description,
            "features": {
                "ipc": ["低延迟", "无需网络", "单进程"],
                "http": ["远程部署", "GPU 加速", "负载均衡", "批处理"]
            }
        }).to_string())
    }
}
