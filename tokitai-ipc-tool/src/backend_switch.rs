//! BackendSwitch - 支持 IPC/HTTP 双后端切换
//!
//! 允许在运行时动态切换本地进程调用和网络调用

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use crate::error::{LidarAiError, Result};

/// IPC 请求结构
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcRequest {
    pub tool: String,
    #[serde(default)]
    pub args: Value,
}

/// IPC 响应结构
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcResponse {
    pub result: Option<Value>,
    pub error: Option<String>,
}

/// API 响应结构（HTTP 模式）
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse {
    pub result: Option<Value>,
    pub error: Option<String>,
}

/// 后端类型枚举
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// 本地 IPC 进程调用
    Ipc,
    /// 网络 HTTP 调用
    Http,
}

/// HTTP 配置
#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

/// IPC 工具运行器
pub struct IpcToolRunner {
    process: Arc<Mutex<Child>>,
}

impl IpcToolRunner {
    /// 启动 Python 工具服务
    pub fn new_python(script_path: &str) -> Result<Self> {
        let child = Command::new("python3")
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(LidarAiError::Io)?;

        Ok(Self {
            process: Arc::new(Mutex::new(child)),
        })
    }

    /// 启动 C++ 工具服务
    pub fn new_cpp(binary_path: &str) -> Result<Self> {
        let child = Command::new(binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(LidarAiError::Io)?;

        Ok(Self {
            process: Arc::new(Mutex::new(child)),
        })
    }

    /// 调用工具
    pub fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
        let mut process = self.process.lock().map_err(|e| {
            LidarAiError::IpcCommunication(format!("锁获取失败：{}", e))
        })?;

        let stdin = process.stdin.as_mut().ok_or_else(|| {
            LidarAiError::IpcCommunication("stdin 不可用".to_string())
        })?;

        // 发送请求
        let request = IpcRequest {
            tool: tool_name.to_string(),
            args,
        };
        let request_json = serde_json::to_string(&request)?;
        writeln!(stdin, "{}", request_json)?;
        stdin.flush()?;

        // 读取响应
        let stdout = process.stdout.as_mut().ok_or_else(|| {
            LidarAiError::IpcCommunication("stdout 不可用".to_string())
        })?;

        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();
        reader.read_line(&mut response_line)?;

        let response: IpcResponse = serde_json::from_str(&response_line)?;

        if let Some(error) = response.error {
            return Err(LidarAiError::ToolExecution(error));
        }

        Ok(response.result.unwrap_or(Value::Null))
    }
}

/// 后端切换器 - 核心组件
#[derive(Clone)]
pub struct BackendSwitch {
    inner: Arc<Mutex<BackendSwitchInner>>,
}

struct BackendSwitchInner {
    backend_type: BackendType,
    ipc_runner: Option<IpcToolRunner>,
    http_client: Option<reqwest::Client>,
    http_config: Option<HttpConfig>,
}

impl BackendSwitch {
    /// 创建 IPC 后端
    pub fn new_ipc(script_path: &str) -> Result<Self> {
        let runner = IpcToolRunner::new_python(script_path)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(BackendSwitchInner {
                backend_type: BackendType::Ipc,
                ipc_runner: Some(runner),
                http_client: None,
                http_config: None,
            })),
        })
    }

    /// 创建 HTTP 后端
    pub fn new_http(base_url: &str, api_key: Option<String>, timeout_secs: u64) -> Self {
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(timeout_secs))
                    .build()
                    .unwrap_or_default()
            })
        });
        let client = handle.join().unwrap();

        Self {
            inner: Arc::new(Mutex::new(BackendSwitchInner {
                backend_type: BackendType::Http,
                ipc_runner: None,
                http_client: Some(client),
                http_config: Some(HttpConfig {
                    base_url: base_url.to_string(),
                    api_key,
                    timeout_secs,
                }),
            })),
        }
    }

    /// 动态切换到 IPC 后端
    pub fn switch_to_ipc(&mut self, script_path: &str) -> Result<()> {
        let runner = IpcToolRunner::new_python(script_path)?;
        let mut inner = self.inner.lock().map_err(|e| {
            LidarAiError::IpcCommunication(format!("锁获取失败：{}", e))
        })?;
        inner.ipc_runner = Some(runner);
        inner.backend_type = BackendType::Ipc;
        tracing::info!("BackendSwitch: 已切换到 IPC 后端：{}", script_path);
        Ok(())
    }

    /// 动态切换到 HTTP 后端
    pub fn switch_to_http(&mut self, base_url: &str, api_key: Option<String>, timeout_secs: u64) {
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(timeout_secs))
                    .build()
                    .unwrap_or_default()
            })
        });
        let client = handle.join().unwrap();

        let mut inner = self.inner.lock().unwrap();
        inner.http_client = Some(client);
        inner.http_config = Some(HttpConfig {
            base_url: base_url.to_string(),
            api_key,
            timeout_secs,
        });
        inner.backend_type = BackendType::Http;
        tracing::info!("BackendSwitch: 已切换到 HTTP 后端：{}", base_url);
    }

    /// 获取当前后端类型
    pub fn current_backend(&self) -> BackendType {
        let inner = self.inner.lock().unwrap();
        inner.backend_type.clone()
    }

    /// 通用工具调用（自动路由到对应后端）
    pub fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
        let inner = self.inner.lock().unwrap();

        match &inner.backend_type {
            BackendType::Ipc => {
                let runner = inner.ipc_runner.as_ref()
                    .ok_or_else(|| LidarAiError::IpcCommunication("IPC runner not initialized".to_string()))?;
                runner.call_tool(tool_name, args)
            }
            BackendType::Http => {
                let client = inner.http_client.as_ref()
                    .ok_or_else(|| LidarAiError::Http("HTTP client not initialized".to_string()))?;
                let config = inner.http_config.as_ref()
                    .ok_or_else(|| LidarAiError::Config("HTTP config not initialized".to_string()))?;

                Self::call_http_tool(client, config, tool_name, args)
            }
        }
    }

    /// HTTP 工具调用实现
    fn call_http_tool(
        client: &reqwest::Client,
        config: &HttpConfig,
        tool_name: &str,
        args: Value,
    ) -> Result<Value> {
        let url = format!("{}/api/v1/{}", config.base_url, tool_name);
        let api_key = config.api_key.clone();
        let client = client.clone();

        // 使用 std::thread::spawn 在独立线程中运行异步 HTTP 请求
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let mut request_builder = client.post(&url).json(&json!({
                    "args": args
                }));

                // 添加 API Key（如果有）
                request_builder = if let Some(ref api_key) = api_key {
                    request_builder.header("Authorization", format!("Bearer {}", api_key))
                } else {
                    request_builder
                };

                let response = request_builder.send().await
                    .map_err(|e| LidarAiError::Http(format!("HTTP 请求失败：{}", e)))?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(LidarAiError::Http(format!("HTTP {}: {}", status, error_text)));
                }

                // 解析响应
                let api_response: ApiResponse = response.json().await
                    .map_err(|e| LidarAiError::Http(format!("JSON 解析失败：{}", e)))?;

                if let Some(error) = api_response.error {
                    return Err(LidarAiError::ToolExecution(error));
                }

                Ok(api_response.result.unwrap_or(Value::Null))
            })
        });

        handle.join().unwrap()
    }
}
