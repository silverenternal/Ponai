//! 错误类型定义

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LidarAiError {
    #[error("IO 错误：{0}")]
    Io(#[from] std::io::Error),

    #[error("JSON 解析错误：{0}")]
    Json(#[from] serde_json::Error),

    #[error("IPC 通信错误：{0}")]
    IpcCommunication(String),

    #[error("工具执行错误：{0}")]
    ToolExecution(String),

    #[error("AI 调度错误：{0}")]
    AiScheduler(String),

    #[error("HTTP 请求错误：{0}")]
    Http(#[from] reqwest::Error),

    #[error("配置错误：{0}")]
    Config(String),

    #[error("Tokitai 工具错误：{0}")]
    Tool(#[from] tokitai::ToolError),
}

pub type Result<T> = std::result::Result<T, LidarAiError>;
