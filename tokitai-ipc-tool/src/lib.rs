//! 3D 点云车机应用 - AI 调度与跨语言工具框架
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        AI 调度层                                 │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
//! │  │  Ollama     │  │  工具路由   │  │  会话管理   │              │
//! │  │  适配器     │  │             │  │             │              │
//! │  └─────────────┘  └─────────────┘  └─────────────┘              │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      tokitai 工具层                             │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
//! │  │  点云工具   │  │  可视化工具 │  │  分析工具   │              │
//! │  │  (IPC)      │  │  (IPC)      │  │  (IPC)      │              │
//! │  └─────────────┘  └─────────────┘  └─────────────┘              │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!              ┌───────────────┼───────────────┐
//!              ▼               ▼               ▼
//! ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//! │  Python 工具    │ │   C++ 工具      │ │   其他语言      │
//! │  (Open3D,       │ │   (PCL,         │ │   (Node.js,    │
//! │   NumPy)        │ │    CUDA)        │ │    Rust)        │
//! └─────────────────┘ └─────────────────┘ └─────────────────┘
//! ```
//!
//! # 后端切换
//!
//! 框架支持 IPC 和 HTTP 两种后端模式，可通过 `BackendSwitch` 动态切换：
//!
//! ```rust,no_run
//! use lidar_ai_studio::BackendSwitch;
//!
//! // 创建 IPC 后端（需要有效的脚本路径）
//! // let switch = BackendSwitch::new_ipc("python_tools/tool.py")?;
//!
//! // 创建 HTTP 后端
//! let switch = BackendSwitch::new_http("http://localhost:8080", None, 30);
//!
//! // 调用工具
//! // let result = switch.call_tool("process", serde_json::json!({}));
//!
//! // 切换到 HTTP 后端
//! // let mut switch = switch;
//! // switch.switch_to_http("http://localhost:8080", None, 30)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # 路径解析
//!
//! 使用 `PathResolver` 解析资源文件路径，支持多种部署场景：
//!
//! ```rust
//! use lidar_ai_studio::path_utils::PathResolver;
//!
//! // 获取相对于可执行文件或当前工作目录的路径
//! match PathResolver::resolve_relative("Cargo.toml") {
//!     Some(path) => println!("Found: {:?}", path),
//!     None => println!("Path not found"),
//! }
//! ```

pub mod ai_scheduler;
pub mod backend;
pub mod backend_switch;
pub mod error;
pub mod instance_seg_tools;
pub mod ipc;
pub mod ipc_error;
pub mod ipc_types;
pub mod path_utils;
pub mod pointcloud_tools;
pub mod tools;

pub use ai_scheduler::{AiScheduler, AiSchedulerConfig};
pub use backend::{Backend, BackendType, HttpBackend, HttpConfig, IpcBackend};
pub use backend_switch::BackendSwitch;
pub use error::{LidarAiError, Result};
pub use instance_seg_tools::InstanceSegToolManager;
pub use ipc_error::{ErrorCode, IpcError};
pub use path_utils::PathResolver;
pub use pointcloud_tools::PointCloudToolManager;
