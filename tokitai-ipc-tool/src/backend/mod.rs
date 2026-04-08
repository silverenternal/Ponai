//! 后端模块 - 支持 IPC/HTTP 双后端
//!
//! 使用策略模式，允许动态切换不同的后端实现

pub mod ipc_backend;
pub mod http_backend;
pub mod trait_def;

pub use ipc_backend::IpcBackend;
pub use http_backend::HttpBackend;
pub use trait_def::{Backend, BackendType, HttpConfig};
