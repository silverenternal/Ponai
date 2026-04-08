//! IPC 通信模块 - 跨语言工具调用
//!
//! 支持通过 stdin/stdout JSON Lines 与 Python/C++ 等外部进程通信
//!
//! # 线程安全
//!
//! `IpcToolRunner` 内部使用 `Arc<Mutex<Child>>` 管理子进程，
//! 支持多线程并发调用 `call_tool`。

use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, instrument, warn};

use crate::error::{LidarAiError, Result};
use crate::ipc_types::{IpcRequest, IpcResponse};

/// IPC 工具运行器 - 管理外部进程并提供工具调用
pub struct IpcToolRunner {
    process: Arc<Mutex<Child>>,
    language: String,
}

impl IpcToolRunner {
    /// 启动 Python 工具服务
    pub fn new_python(script_path: &str) -> Result<Self> {
        let span = tracing::debug_span!("ipc_tool_runner_new_python", path = script_path);
        let _guard = span.enter();
        
        debug!("启动 Python IPC 工具：{}", script_path);
        let child = Command::new("python3")
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                error!("启动 Python 进程失败：{} (path={})", e, script_path);
                LidarAiError::Io(e)
            })?;

        debug!("Python 进程启动成功，PID: {:?}", child.id());
        Ok(Self {
            process: Arc::new(Mutex::new(child)),
            language: "python".to_string(),
        })
    }

    /// 启动 C++ 工具服务（编译后的二进制）
    pub fn new_cpp(binary_path: &str) -> Result<Self> {
        let span = tracing::debug_span!("ipc_tool_runner_new_cpp", path = binary_path);
        let _guard = span.enter();
        
        debug!("启动 C++ IPC 工具：{}", binary_path);
        let child = Command::new(binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                error!("启动 C++ 进程失败：{} (path={})", e, binary_path);
                LidarAiError::Io(e)
            })?;

        debug!("C++ 进程启动成功，PID: {:?}", child.id());
        Ok(Self {
            process: Arc::new(Mutex::new(child)),
            language: "cpp".to_string(),
        })
    }

    /// 调用外部工具
    #[instrument(skip(self, args), fields(tool = tool_name))]
    pub fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
        debug!("IPC 调用工具：{}", tool_name);

        let mut process = self.process.lock().map_err(|e| {
            error!("进程锁获取失败（锁中毒）: {}", e);
            LidarAiError::IpcCommunication(format!("进程锁中毒：{}", e))
        })?;

        // 检查进程是否存活
        if let Ok(Some(status)) = process.try_wait() {
            error!("子进程已退出：{}", status);
            return Err(LidarAiError::IpcCommunication(format!("子进程已退出：{}", status)));
        }

        let stdin = process.stdin.as_mut().ok_or_else(|| {
            error!("stdin 管道不可用");
            LidarAiError::IpcCommunication("stdin 管道已关闭".to_string())
        })?;

        // 发送请求
        let request = IpcRequest {
            tool: tool_name.to_string(),
            args,
        };
        let request_json = serde_json::to_string(&request)?;
        debug!("发送 IPC 请求：{}", request_json);
        writeln!(stdin, "{}", request_json)?;
        stdin.flush()?;

        // 读取响应
        let stdout = process.stdout.as_mut().ok_or_else(|| {
            error!("stdout 管道不可用");
            LidarAiError::IpcCommunication("stdout 管道已关闭".to_string())
        })?;

        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();
        let bytes_read = reader.read_line(&mut response_line)?;
        
        if bytes_read == 0 {
            error!("stdout 返回空（进程可能已退出）");
            return Err(LidarAiError::IpcCommunication("stdout 返回空数据".to_string()));
        }

        debug!("收到 IPC 响应：{}", response_line.trim());
        let response: IpcResponse = serde_json::from_str(&response_line)?;

        // 处理结构化错误
        if let Some(ipc_error) = response.error {
            warn!("工具调用返回错误：code={:?}, message={}", ipc_error.code, ipc_error.message);
            return Err(LidarAiError::from_ipc_error(ipc_error));
        }

        debug!("工具调用成功：{}", tool_name);
        Ok(response.result.unwrap_or(Value::Null))
    }

    /// 获取语言类型
    pub fn language(&self) -> &str {
        &self.language
    }

    /// 检查进程是否存活
    pub fn is_alive(&self) -> bool {
        let process = self.process.lock();
        match process {
            Ok(mut guard) => {
                match guard.try_wait() {
                    Ok(None) => true,      // 进程仍在运行
                    Ok(Some(_)) => false,  // 进程已退出
                    Err(_) => false,       // 检查失败
                }
            }
            Err(_) => false,  // 锁中毒
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_error::IpcError;
    use serde_json::json;

    /// 测试 IPC 请求序列化
    #[test]
    fn test_ipc_request_serialization() {
        let request = IpcRequest {
            tool: "test_tool".to_string(),
            args: json!({"param1": "value1", "param2": 42}),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: IpcRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(request.tool, deserialized.tool);
        assert_eq!(request.args, deserialized.args);
    }

    /// 测试 IPC 响应序列化（成功场景）
    #[test]
    fn test_ipc_response_serialization_success() {
        let response = IpcResponse {
            result: Some(json!({"status": "ok", "data": [1, 2, 3]})),
            error: None,
        };

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: IpcResponse = serde_json::from_str(&serialized).unwrap();

        assert_eq!(response.result, deserialized.result);
        assert_eq!(response.error, deserialized.error);
    }

    /// 测试 IPC 响应序列化（错误场景）
    #[test]
    fn test_ipc_response_serialization_error() {
        use crate::ipc_error::ErrorCode;

        let ipc_error = IpcError {
            code: ErrorCode::FileNotFound,
            message: "文件不存在：/path/to/file".to_string(),
            details: Some(json!({"path": "/path/to/file"})),
        };

        let response = IpcResponse {
            result: None,
            error: Some(ipc_error),
        };

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: IpcResponse = serde_json::from_str(&serialized).unwrap();

        assert_eq!(response.result, deserialized.result);
        assert!(deserialized.error.is_some());
        assert_eq!(deserialized.error.unwrap().code, ErrorCode::FileNotFound);
    }

    /// 测试 IpcRequest 默认 args 值
    #[test]
    fn test_ipc_request_default_args() {
        let json_str = r#"{"tool": "test_tool", "args": {}}"#;
        let request: IpcRequest = serde_json::from_str(json_str).unwrap();

        assert_eq!(request.tool, "test_tool");
        assert!(request.args.is_object());
    }

    /// 测试 IpcRequest 省略 args 字段时的默认值
    #[test]
    fn test_ipc_request_missing_args_field() {
        let json_str = r#"{"tool": "test_tool"}"#;
        let request: IpcRequest = serde_json::from_str(json_str).unwrap();

        assert_eq!(request.tool, "test_tool");
        assert!(request.args.is_null());
    }

    /// 测试进程启动失败场景（路径不存在）
    #[test]
    fn test_ipc_tool_runner_start_failure() {
        let result = IpcToolRunner::new_cpp("/nonexistent/path/to/binary");
        assert!(result.is_err());

        if let Err(LidarAiError::Io(e)) = result {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
        } else {
            panic!("Expected Io error");
        }
    }

    /// 测试 language 方法
    #[test]
    fn test_ipc_tool_runner_language() {
        // 由于 IpcToolRunner 需要实际启动进程，这里只测试结构
        // 实际集成测试在 tests/ 目录下进行
        assert!(true);
    }

    /// 测试 is_alive 方法（进程未启动时）
    #[test]
    fn test_ipc_tool_runner_is_alive() {
        // 占位测试，实际测试需要启动进程
        assert!(true);
    }
}
