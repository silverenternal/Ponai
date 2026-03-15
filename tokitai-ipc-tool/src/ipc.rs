//! IPC 通信模块 - 跨语言工具调用
//!
//! 支持通过 stdin/stdout JSON Lines 与 Python/C++ 等外部进程通信

use serde::{Deserialize, Serialize};
use serde_json::Value;
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

/// IPC 工具运行器 - 管理外部进程并提供工具调用
pub struct IpcToolRunner {
    process: Arc<Mutex<Child>>,
    language: String,
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
            .map_err(|e| LidarAiError::Io(e))?;

        Ok(Self {
            process: Arc::new(Mutex::new(child)),
            language: "python".to_string(),
        })
    }

    /// 启动 C++ 工具服务（编译后的二进制）
    pub fn new_cpp(binary_path: &str) -> Result<Self> {
        let child = Command::new(binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| LidarAiError::Io(e))?;

        Ok(Self {
            process: Arc::new(Mutex::new(child)),
            language: "cpp".to_string(),
        })
    }

    /// 调用外部工具
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

    /// 获取语言类型
    pub fn language(&self) -> &str {
        &self.language
    }
}
