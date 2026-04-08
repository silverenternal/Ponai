//! 点云工具层 - 包装 Python/C++ 点云处理工具
//!
//! 使用 tokitai 宏将点云工具暴露给 AI 调用

use serde_json::{json, Value};
use tokitai::tool;
use tokitai::ToolError;

use crate::error::LidarAiError;
use crate::ipc::IpcToolRunner;
use crate::ipc_error::ErrorCode;

/// 点云工具管理器
pub struct PointCloudToolManager {
    python_runner: Option<IpcToolRunner>,
    cpp_runner: Option<IpcToolRunner>,
}

impl PointCloudToolManager {
    /// 创建新的工具管理器（仅 Python）
    pub fn new_python(python_script: &str) -> Result<Self, LidarAiError> {
        let runner = IpcToolRunner::new_python(python_script)?;
        Ok(Self {
            python_runner: Some(runner),
            cpp_runner: None,
        })
    }

    /// 创建新的工具管理器（Python + C++）
    pub fn new_dual(python_script: &str, cpp_binary: &str) -> Result<Self, LidarAiError> {
        let python_runner = IpcToolRunner::new_python(python_script)?;
        let cpp_runner = IpcToolRunner::new_cpp(cpp_binary)?;
        Ok(Self {
            python_runner: Some(python_runner),
            cpp_runner: Some(cpp_runner),
        })
    }

    /// 获取 Python runner
    fn python(&self) -> Result<&IpcToolRunner, LidarAiError> {
        self.python_runner
            .as_ref()
            .ok_or_else(|| LidarAiError::ToolExecution {
                code: ErrorCode::InternalError,
                message: "Python 工具未初始化".to_string(),
            })
    }

    /// 获取 C++ runner
    fn cpp(&self) -> Result<&IpcToolRunner, LidarAiError> {
        self.cpp_runner
            .as_ref()
            .ok_or_else(|| LidarAiError::ToolExecution {
                code: ErrorCode::InternalError,
                message: "C++ 工具未初始化".to_string(),
            })
    }

    /// 通用工具调用方法
    fn invoke_tool(&self, backend: &str, tool_name: &str, args: Value) -> Result<Value, LidarAiError> {
        match backend {
            "python" => self.python()?.call_tool(tool_name, args),
            "cpp" => self.cpp()?.call_tool(tool_name, args),
            _ => Err(LidarAiError::ToolExecution {
                code: ErrorCode::InvalidParameter,
                message: format!("未知的后端：{}", backend),
            }),
        }
    }
}

#[tool]
impl PointCloudToolManager {
    /// 加载点云文件（支持 PCD、PLY、LAS 格式）
    pub fn load_point_cloud(&self, file_path: String) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "load_point_cloud", json!({
            "file_path": file_path
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 获取点云基本信息（点数、边界框、密度等）
    pub fn get_point_cloud_info(&self) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "get_point_cloud_info", json!({})).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(serde_json::to_string_pretty(&result).map_err(|e| ToolError::validation_error(e.to_string()))?)
    }

    /// 点云降采样（体素网格滤波）
    pub fn downsample(&self, voxel_size: f64) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "downsample", json!({
            "voxel_size": voxel_size
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 法线估计
    pub fn estimate_normals(&self, k_neighbors: u32) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "estimate_normals", json!({
            "k_neighbors": k_neighbors
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 离群点移除
    pub fn remove_outliers(&self, nb_neighbors: u32, std_ratio: f64) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "remove_outliers", json!({
            "nb_neighbors": nb_neighbors,
            "std_ratio": std_ratio
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 平面分割（RANSAC）
    pub fn segment_plane(&self, distance_threshold: f64, max_iterations: u32) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "segment_plane", json!({
            "distance_threshold": distance_threshold,
            "max_iterations": max_iterations
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 欧式聚类
    pub fn euclidean_clustering(&self, tolerance: f64, min_cluster_size: u32, max_cluster_size: u32) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "euclidean_clustering", json!({
            "tolerance": tolerance,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": max_cluster_size
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 保存点云到文件
    pub fn save_point_cloud(&self, file_path: String) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "save_point_cloud", json!({
            "file_path": file_path
        })).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 可视化点云（打开可视化窗口）
    pub fn visualize(&self) -> Result<String, ToolError> {
        let result = self.invoke_tool("python", "visualize", json!({})).map_err(|e| ToolError::validation_error(e.to_string()))?;
        Ok(result["message"].as_str().unwrap_or("").to_string())
    }

    /// 获取可用的 C++ 高性能工具列表
    pub fn get_cpp_tools(&self) -> Result<String, ToolError> {
        // 如果 C++ 后端可用，返回工具列表
        match self.cpp() {
            Ok(_) => Ok(json!({
                "available": true,
                "tools": ["gpu_accelerated_filter", "real_time_segmentation", "cuda_normals"]
            }).to_string()),
            Err(_) => Ok(json!({
                "available": false,
                "reason": "C++ 后端未启用"
            }).to_string()),
        }
    }
}
