//! IPC 后端实现
//!
//! 通过 stdin/stdout JSON Lines 与 Python/C++ 进程通信
//!
//! # 线程安全设计
//!
//! `IpcBackend` 内部使用 `Arc<Mutex<Child>>` 管理子进程，
//! 确保多线程环境下的安全访问。调用 `call_tool` 时会自动加锁，
//! 串行化对同一进程的请求。

use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tracing::{info, debug, warn, error, instrument};

use crate::error::{LidarAiError, Result};
use crate::backend::trait_def::Backend;
use crate::ipc_types::{IpcRequest, IpcResponse};

/// IPC 后端 - 管理外部进程并提供工具调用
///
/// # 线程安全
///
/// 多个线程可以同时调用同一个 `IpcBackend` 实例的 `call_tool` 方法，
/// 内部互斥锁会保证同一时间只有一个请求在使用进程管道。
///
/// # 示例
///
/// ```rust,no_run
/// use lidar_ai_studio::backend::{Backend, IpcBackend};
///
/// let backend = IpcBackend::new_python("python_tools/tool.py").unwrap();
/// let result = backend.call_tool("process", serde_json::json!({"key": "value"}));
/// ```
pub struct IpcBackend {
    process: Arc<Mutex<Child>>,
}

impl IpcBackend {
    /// 启动 Python 工具服务
    ///
    /// # 参数
    /// - `script_path`: Python 脚本路径（绝对或相对当前工作目录）
    ///
    /// # 返回
    /// - `Ok(IpcBackend)`: 成功启动进程
    /// - `Err(LidarAiError::Io)`: 进程启动失败
    pub fn new_python(script_path: &str) -> Result<Self> {
        let span = tracing::debug_span!("ipc_new_python", path = script_path);
        let _guard = span.enter();
        
        info!("启动 Python IPC 服务：{}", script_path);
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
        })
    }

    /// 启动 C++ 工具服务
    ///
    /// # 参数
    /// - `binary_path`: 可执行文件路径
    pub fn new_cpp(binary_path: &str) -> Result<Self> {
        let span = tracing::debug_span!("ipc_new_cpp", path = binary_path);
        let _guard = span.enter();
        
        info!("启动 C++ IPC 服务：{}", binary_path);
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
        })
    }

    /// 检查子进程是否存活
    fn process_alive(&self) -> bool {
        let process = self.process.lock();
        match process {
            Ok(mut guard) => {
                // try_wait 返回 Ok(Some(status)) 表示进程已退出
                // 返回 Ok(None) 表示进程仍在运行
                match guard.try_wait() {
                    Ok(None) => true,      // 进程仍在运行
                    Ok(Some(_)) => false,  // 进程已退出
                    Err(_) => false,       // 检查失败，假设不可用
                }
            }
            Err(_) => false,  // 锁中毒，假设不可用
        }
    }
}

impl Backend for IpcBackend {
    #[instrument(skip(self, args), fields(tool = tool_name))]
    fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
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

        // 构建并发送请求
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

    fn backend_type(&self) -> &'static str {
        "ipc"
    }

    fn is_available(&self) -> bool {
        self.process_alive()
    }
}
