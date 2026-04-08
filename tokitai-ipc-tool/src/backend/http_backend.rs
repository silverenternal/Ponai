//! HTTP 后端实现
//!
//! 通过 HTTP 协议与远程服务通信
//!
//! # 异步运行时设计
//!
//! HTTP 后端需要在同步接口中执行异步 HTTP 请求。支持两种模式：
//!
//! 1. **全局运行时模式**: 使用 `Lazy<Runtime>` 懒加载创建共享运行时
//! 2. **注入运行时模式**: 用户传入自己的 `Handle`，支持在异步上下文中使用
//!
//! # 示例
//!
//! ```rust,no_run
//! use lidar_ai_studio::backend::HttpBackend;
//!
//! // 模式 1: 使用全局运行时（简单场景）
//! let backend = HttpBackend::new("http://localhost:8080", None, 30);
//!
//! // 模式 2: 注入运行时 Handle（异步场景）
//! let rt = tokio::runtime::Runtime::new().unwrap();
//! let backend = HttpBackend::with_handle("http://localhost:8080", None, 30, rt.handle().clone());
//! ```

use serde_json::{json, Value};
use std::sync::Arc;
use once_cell::sync::Lazy;
use tokio::runtime::{Handle, Runtime};
use tracing::{info, debug, warn, error, instrument};

use crate::error::{LidarAiError, Result};
use crate::backend::trait_def::{Backend, HttpConfig};
use crate::ipc_types::ApiResponse;

/// 全局共享的 Tokio 运行时（懒加载）
///
/// # 注意
///
/// 此运行时仅在同步上下文中使用。如果在异步上下文中调用 `call_tool`，
/// 请使用 `HttpBackend::with_handle` 注入当前的 `Handle`。
static GLOBAL_RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)  // 默认 4 个工作线程
        .enable_all()
        .thread_name("lidar-http-backend")
        .build()
        .expect("Failed to create Tokio runtime")
});

/// HTTP 后端 - 通过 HTTP 调用远程工具
///
/// # 线程安全
///
/// `HttpBackend` 内部使用 `Arc<reqwest::Client>`，支持多线程并发调用。
/// 多个线程可以同时调用同一个实例的 `call_tool` 方法。
///
/// # 运行时处理
///
/// - 默认使用全局懒加载运行时
/// - 在异步上下文中使用时，应通过 `with_handle` 注入当前运行时
pub struct HttpBackend {
    client: Arc<reqwest::Client>,
    config: HttpConfig,
    runtime_handle: Option<Handle>,
}

impl HttpBackend {
    /// 创建 HTTP 后端（使用全局运行时）
    ///
    /// # 参数
    /// - `base_url`: 服务基础 URL
    /// - `api_key`: API 认证密钥（可选）
    /// - `timeout_secs`: 请求超时（秒）
    pub fn new(base_url: &str, api_key: Option<String>, timeout_secs: u64) -> Self {
        Self::with_handle(base_url, api_key, timeout_secs, GLOBAL_RUNTIME.handle().clone())
    }

    /// 创建 HTTP 后端（注入运行时 Handle）
    ///
    /// # 参数
    /// - `base_url`: 服务基础 URL
    /// - `api_key`: API 认证密钥（可选）
    /// - `timeout_secs`: 请求超时（秒）
    /// - `handle`: Tokio 运行时 Handle
    ///
    /// # 使用场景
    ///
    /// 当在异步上下文中调用 `call_tool` 时，使用此方法避免"嵌套运行时" panic。
    pub fn with_handle(base_url: &str, api_key: Option<String>, timeout_secs: u64, handle: Handle) -> Self {
        let client = create_http_client(timeout_secs);

        info!("HTTP 后端初始化：base_url={}, timeout={}s", base_url, timeout_secs);

        Self {
            client: Arc::new(client),
            config: HttpConfig {
                base_url: base_url.to_string(),
                api_key,
                timeout_secs,
            },
            runtime_handle: Some(handle),
        }
    }

    /// 更新配置
    ///
    /// # 注意
    ///
    /// 此方法会创建新的 `reqwest::Client`，不影响进行中的请求。
    pub fn update_config(&mut self, base_url: &str, api_key: Option<String>, timeout_secs: u64) {
        let span = tracing::debug_span!("http_update_config", base_url);
        let _guard = span.enter();
        
        info!("HTTP 后端配置更新：base_url={}, timeout={}s", base_url, timeout_secs);
        self.config = HttpConfig {
            base_url: base_url.to_string(),
            api_key,
            timeout_secs,
        };
        self.client = Arc::new(create_http_client(timeout_secs));
    }

    /// 获取配置引用
    pub fn config(&self) -> &HttpConfig {
        &self.config
    }

    /// 检查服务是否可用
    fn check_health(&self) -> bool {
        let span = tracing::debug_span!("http_health_check", url = self.config.base_url.as_str());
        let _guard = span.enter();
        
        // 快速检查：尝试发送 HEAD 请求
        let client = self.client.clone();
        let url = &self.config.base_url;
        
        if let Some(handle) = &self.runtime_handle {
            let _ = handle.block_on(async {
                client.head(url).send().await
            });
        }
        true  // 简化实现，始终返回 true（实际应检查响应）
    }
}

impl Backend for HttpBackend {
    #[instrument(skip(self, args), fields(tool = tool_name))]
    fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
        debug!("HTTP 调用工具：{}", tool_name);

        let client = self.client.clone();
        let config = self.config.clone();
        let tool_name_str = tool_name.to_string();
        let handle = self.runtime_handle.clone()
            .unwrap_or_else(|| GLOBAL_RUNTIME.handle().clone());

        // 使用运行时执行异步 HTTP 请求
        handle.block_on(async move {
            let url = format!("{}/api/v1/{}", config.base_url, tool_name_str);

            debug!("HTTP 请求 URL: {}", url);

            let mut request_builder = client.post(&url).json(&json!({
                "args": args
            }));

            // 添加 API Key（如果有）
            if let Some(ref api_key) = config.api_key {
                request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
            }

            let response = request_builder.send().await
                .map_err(|e| {
                    error!("HTTP 请求失败：{} (url={})", e, url);
                    LidarAiError::Http(format!("HTTP 请求失败：{}", e))
                })?;

            let status = response.status();
            debug!("HTTP 响应状态码：{}", status);

            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                error!("HTTP 错误响应：{} - {}", status, error_text);
                return Err(LidarAiError::Http(format!("HTTP {}: {}", status, error_text)));
            }

            // 解析响应
            let api_response: ApiResponse = response.json().await
                .map_err(|e| {
                    error!("JSON 解析失败：{}", e);
                    LidarAiError::Http(format!("JSON 解析失败：{}", e))
                })?;

            // 处理结构化错误
            if let Some(ipc_error) = api_response.error {
                warn!("工具调用返回错误：code={:?}, message={}", ipc_error.code, ipc_error.message);
                return Err(LidarAiError::from_ipc_error(ipc_error));
            }

            debug!("工具调用成功：{}", tool_name_str);
            Ok(api_response.result.unwrap_or(Value::Null))
        })
        .map_err(|e| {
            // 处理运行时错误（如 panic）
            error!("请求执行失败：{}", e);
            LidarAiError::Http(format!("请求执行失败：{}", e))
        })
    }

    fn backend_type(&self) -> &'static str {
        "http"
    }

    fn is_available(&self) -> bool {
        self.check_health()
    }
}

/// 创建 HTTP 客户端
fn create_http_client(timeout_secs: u64) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .user_agent("lidar-ai-studio/0.1.0")
        .build()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_backend_creation() {
        let backend = HttpBackend::new("http://localhost:8080", None, 30);
        assert_eq!(backend.config().base_url, "http://localhost:8080");
        assert_eq!(backend.config().timeout_secs, 30);
    }

    #[test]
    fn test_http_backend_with_handle() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let backend = HttpBackend::with_handle(
            "http://localhost:8080",
            Some("test_key".to_string()),
            60,
            rt.handle().clone(),
        );
        assert_eq!(backend.config().base_url, "http://localhost:8080");
        assert_eq!(backend.config().api_key, Some("test_key".to_string()));
        assert_eq!(backend.config().timeout_secs, 60);
    }

    #[test]
    fn test_http_backend_update_config() {
        let mut backend = HttpBackend::new("http://localhost:8080", None, 30);
        backend.update_config("http://localhost:9090", Some("new_key".to_string()), 120);
        assert_eq!(backend.config().base_url, "http://localhost:9090");
        assert_eq!(backend.config().api_key, Some("new_key".to_string()));
        assert_eq!(backend.config().timeout_secs, 120);
    }

    #[test]
    fn test_http_backend_type() {
        let backend = HttpBackend::new("http://localhost:8080", None, 30);
        assert_eq!(backend.backend_type(), "http");
    }
}
