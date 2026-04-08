//! 后端 Trait 定义
//!
//! 定义统一的 Backend 接口，支持 IPC/HTTP 等多种后端实现
//!
//! # 设计说明
//!
//! `Backend` trait 采用对象安全设计，支持 `dyn Backend` 动态分发。
//! 每个后端实现内部管理自己的同步原语（如 `Arc<Mutex<Child>>`），
//! 避免外层嵌套多层锁导致的性能瓶颈。

use serde_json::Value;

use crate::error::Result;

/// 后端 Trait - 所有后端实现必须实现此接口
///
/// # 设计原则
///
/// - **对象安全**: 支持 `dyn Backend` 动态分发
/// - **自管理同步**: 每个后端内部管理自己的锁，避免外层嵌套
/// - **无状态切换**: 后端实例创建后配置不可变，切换需创建新实例
pub trait Backend: Send + Sync {
    /// 调用工具
    ///
    /// # 参数
    /// - `tool_name`: 工具名称
    /// - `args`: JSON 格式参数
    ///
    /// # 返回值
    /// - `Ok(Value)`: 工具执行结果
    /// - `Err(LidarAiError)`: 执行失败
    fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value>;

    /// 获取后端类型标识
    fn backend_type(&self) -> &'static str;

    /// 检查后端健康状态
    ///
    /// # 实现说明
    /// - IPC 后端：检查子进程是否存活
    /// - HTTP 后端：发送 HEAD 请求检查服务可用性
    fn is_available(&self) -> bool;
}

/// 后端类型枚举 - 用于配置和序列化
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum BackendType {
    /// 本地 IPC 进程调用
    Ipc,
    /// 网络 HTTP 调用
    Http,
}

/// HTTP 后端配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HttpConfig {
    /// 基础 URL
    pub base_url: String,
    /// API 认证密钥（可选）
    pub api_key: Option<String>,
    /// 请求超时（秒）
    pub timeout_secs: u64,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".to_string(),
            api_key: None,
            timeout_secs: 30,
        }
    }
}
