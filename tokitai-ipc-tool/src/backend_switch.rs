//! BackendSwitch - 支持 IPC/HTTP 双后端切换
//!
//! 使用策略模式，允许在运行时动态切换不同的后端实现
//!
//! # 架构设计
//!
//! `BackendSwitch` 使用 `Arc<dyn Backend>` 直接持有后端实例，
//! 避免多层锁嵌套。每个后端内部管理自己的同步原语。
//!
//! ```text
//! 旧架构:
//! BackendSwitch -> Arc<Mutex<BackendSwitchInner -> Arc<Mutex<Box<dyn Backend>>>>>
//!                  (外层锁)                     (内层锁)
//!
//! 新架构:
//! BackendSwitch -> Arc<RwLock<dyn Backend>>
//!                  (单锁，读写分离)
//! ```
//!
//! # 示例
//!
//! ```rust,no_run
//! # use serde_json::json;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use lidar_ai_studio::BackendSwitch;
//!
//! // 创建 IPC 后端
//! let switch = BackendSwitch::new_ipc("python_tools/pointcloud_tools.py")?;
//!
//! // 调用工具
//! let result = switch.call_tool("process_pointcloud", json!({"path": "test.pcd"}))?;
//!
//! // 切换到 HTTP 后端
//! let mut switch = switch;
//! switch.switch_to_http("http://localhost:8080", None, 30)?;
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, RwLock};
use serde_json::Value;
use tracing::info;

use crate::error::{LidarAiError, Result};
use crate::backend::{Backend, BackendType, IpcBackend, HttpBackend};

/// 后端切换器 - 核心组件
///
/// 使用策略模式，允许在运行时动态切换不同的后端实现
///
/// # 线程安全
///
/// `BackendSwitch` 使用 `Arc<RwLock<Box<dyn Backend>>>` 实现：
/// - 读操作（`call_tool`）：共享锁，支持并发
/// - 写操作（`switch_to_*`）：独占锁，阻塞其他操作
///
/// # 性能说明
///
/// 多个 `BackendSwitch` clone 共享同一个后端实例。
/// 切换后端会影响所有 clone 的实例。
pub struct BackendSwitch {
    inner: Arc<RwLock<BackendSwitchInner>>,
}

struct BackendSwitchInner {
    backend_type: BackendType,
    backend: Box<dyn Backend>,
}

impl BackendSwitch {
    /// 创建 IPC 后端
    ///
    /// # 参数
    /// - `script_path`: Python 脚本路径
    ///
    /// # 返回
    /// - `Ok(BackendSwitch)`: 成功创建
    /// - `Err(LidarAiError)`: 进程启动失败
    pub fn new_ipc(script_path: &str) -> Result<Self> {
        let backend = IpcBackend::new_python(script_path)?;
        let boxed_backend: Box<dyn Backend> = Box::new(backend);

        info!("BackendSwitch 创建 IPC 后端：{}", script_path);
        Ok(Self {
            inner: Arc::new(RwLock::new(BackendSwitchInner {
                backend_type: BackendType::Ipc,
                backend: boxed_backend,
            })),
        })
    }

    /// 创建 HTTP 后端
    ///
    /// # 参数
    /// - `base_url`: 服务基础 URL
    /// - `api_key`: API 认证密钥（可选）
    /// - `timeout_secs`: 请求超时（秒）
    pub fn new_http(base_url: &str, api_key: Option<String>, timeout_secs: u64) -> Self {
        let backend = HttpBackend::new(base_url, api_key, timeout_secs);
        let boxed_backend: Box<dyn Backend> = Box::new(backend);

        info!("BackendSwitch 创建 HTTP 后端：{}", base_url);
        Self {
            inner: Arc::new(RwLock::new(BackendSwitchInner {
                backend_type: BackendType::Http,
                backend: boxed_backend,
            })),
        }
    }

    /// 动态切换到 IPC 后端
    ///
    /// # 注意
    ///
    /// 此操作会阻塞所有正在进行的 `call_tool` 调用，
    /// 直到切换完成。
    pub fn switch_to_ipc(&mut self, script_path: &str) -> Result<()> {
        let backend = IpcBackend::new_python(script_path)?;
        let boxed_backend: Box<dyn Backend> = Box::new(backend);

        let mut inner = self.inner.write().map_err(|e| {
            LidarAiError::IpcCommunication(format!("锁获取失败（锁中毒）: {}", e))
        })?;

        inner.backend = boxed_backend;
        inner.backend_type = BackendType::Ipc;
        info!("BackendSwitch: 已切换到 IPC 后端：{}", script_path);
        Ok(())
    }

    /// 动态切换到 HTTP 后端
    pub fn switch_to_http(&mut self, base_url: &str, api_key: Option<String>, timeout_secs: u64) -> Result<()> {
        let backend = HttpBackend::new(base_url, api_key, timeout_secs);
        let boxed_backend: Box<dyn Backend> = Box::new(backend);

        let mut inner = self.inner.write().map_err(|e| {
            LidarAiError::IpcCommunication(format!("锁获取失败（锁中毒）: {}", e))
        })?;

        inner.backend = boxed_backend;
        inner.backend_type = BackendType::Http;
        info!("BackendSwitch: 已切换到 HTTP 后端：{}", base_url);
        Ok(())
    }

    /// 获取当前后端类型
    ///
    /// # 注意
    ///
    /// 此方法使用读锁，不会阻塞其他读操作。
    pub fn current_backend(&self) -> BackendType {
        let inner = self.inner.read().unwrap();
        inner.backend_type.clone()
    }

    /// 通用工具调用（自动路由到对应后端）
    ///
    /// # 参数
    /// - `tool_name`: 工具名称
    /// - `args`: JSON 格式参数
    ///
    /// # 性能
    ///
    /// 多个线程可以同时调用 `call_tool`，
    /// 实际并发度取决于底层后端的实现。
    pub fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
        let inner = self.inner.read().unwrap();
        inner.backend.call_tool(tool_name, args)
    }

    /// 判断当前后端是否可用
    ///
    /// # 实现说明
    ///
    /// - IPC 后端：检查子进程是否存活
    /// - HTTP 后端：发送 HEAD 请求检查服务可用性
    pub fn is_available(&self) -> bool {
        let inner = self.inner.read().unwrap();
        inner.backend.is_available()
    }
}

impl Clone for BackendSwitch {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 BackendType 枚举
    #[test]
    fn test_backend_type_enum() {
        let ipc = BackendType::Ipc;
        let http = BackendType::Http;

        assert_ne!(ipc, http);
        assert_eq!(ipc, BackendType::Ipc);
        assert_eq!(http, BackendType::Http);
    }

    /// 测试 BackendSwitch 创建 HTTP 后端
    #[test]
    fn test_backend_switch_new_http() {
        let backend_switch = BackendSwitch::new_http("http://localhost:8080", None, 30);
        assert_eq!(backend_switch.current_backend(), BackendType::Http);
    }

    /// 测试 BackendSwitch 后端类型查询
    #[test]
    fn test_backend_switch_current_backend() {
        let backend_switch = BackendSwitch::new_http("http://localhost:8080", None, 30);
        let backend_type = backend_switch.current_backend();
        assert_eq!(backend_type, BackendType::Http);
    }

    /// 测试 BackendSwitch 切换到 HTTP 后端
    #[test]
    fn test_backend_switch_switch_to_http() {
        let mut backend_switch = BackendSwitch::new_http("http://localhost:8080", None, 30);

        // 切换到另一个 HTTP 后端
        let result = backend_switch.switch_to_http("http://localhost:9090", Some("new_key".to_string()), 60);
        assert!(result.is_ok());
        assert_eq!(backend_switch.current_backend(), BackendType::Http);
    }

    /// 测试并发安全性（多 Clone 共享状态）
    #[test]
    fn test_backend_switch_clone_shares_state() {
        let backend_switch = BackendSwitch::new_http("http://localhost:8080", None, 30);
        let clone1 = backend_switch.clone();
        let clone2 = backend_switch.clone();

        // 所有 Clone 应该共享同一个后端类型
        assert_eq!(backend_switch.current_backend(), BackendType::Http);
        assert_eq!(clone1.current_backend(), BackendType::Http);
        assert_eq!(clone2.current_backend(), BackendType::Http);
    }

    /// 测试 is_available 方法
    #[test]
    fn test_backend_switch_is_available() {
        let backend_switch = BackendSwitch::new_http("http://localhost:8080", None, 30);
        // HTTP 后端应该始终可用（不检查连接）
        assert!(backend_switch.is_available());
    }

    /// 测试 RwLock 读锁并发（多个线程同时读）
    #[test]
    fn test_rwlock_concurrent_read() {
        let backend_switch = BackendSwitch::new_http("http://localhost:8080", None, 30);
        let mut handles = vec![];

        // 创建 10 个并发读操作
        for _ in 0..10 {
            let switch = backend_switch.clone();
            let handle = std::thread::spawn(move || {
                switch.current_backend()
            });
            handles.push(handle);
        }

        // 所有线程应该都能成功获取读锁
        for handle in handles {
            assert_eq!(handle.join().unwrap(), BackendType::Http);
        }
    }
}
