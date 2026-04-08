//! 错误类型定义
//!
//! # 设计原则
//!
//! - **统一错误处理**: 所有错误转换为 `LidarAiError`
//! - **跨语言兼容**: 支持从 IPC 结构化错误转换
//! - **可恢复性判断**: 提供 `is_recoverable()` 和 `is_server_error()` 方法

use thiserror::Error;

use crate::ipc_error::{IpcError, ErrorCode};

/// 主错误类型
#[derive(Error, Debug)]
pub enum LidarAiError {
    #[error("IO 错误：{0}")]
    Io(#[from] std::io::Error),

    #[error("JSON 解析错误：{0}")]
    Json(#[from] serde_json::Error),

    #[error("IPC 通信错误：{0}")]
    IpcCommunication(String),

    #[error("工具执行错误：{code} - {message}")]
    ToolExecution {
        code: ErrorCode,
        message: String,
    },

    #[error("AI 调度错误：{0}")]
    AiScheduler(String),

    #[error("HTTP 请求错误：{0}")]
    Http(String),

    #[error("配置错误：{0}")]
    Config(String),

    #[error("Tokitai 工具错误：{0}")]
    Tool(#[from] tokitai::ToolError),
}

impl LidarAiError {
    /// 从 IPC 结构化错误转换
    pub fn from_ipc_error(ipc_error: IpcError) -> Self {
        LidarAiError::ToolExecution {
            code: ipc_error.code,
            message: ipc_error.message,
        }
    }

    /// 获取错误码（如果是 ToolExecution 类型）
    pub fn error_code(&self) -> Option<&ErrorCode> {
        match self {
            LidarAiError::ToolExecution { code, .. } => Some(code),
            _ => None,
        }
    }

    /// 判断是否为可恢复错误
    ///
    /// 可恢复错误包括：
    /// - 通信超时、连接断开（可重试）
    /// - 服务暂时不可用（可切换备用服务）
    /// - 请求频率超限（等待后可重试）
    ///
    /// # 注意
    ///
    /// HTTP 错误和 IO 错误默认不可恢复，
    /// 除非它们包装了可恢复的底层错误。
    pub fn is_recoverable(&self) -> bool {
        match self {
            LidarAiError::ToolExecution { code, .. } => code.is_recoverable(),
            // IO 错误中的超时/连接断开可能可恢复
            LidarAiError::Io(e) => {
                matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::UnexpectedEof
                )
            }
            // HTTP 错误中的超时/服务不可用可能可恢复
            LidarAiError::Http(msg) => {
                msg.contains("timeout")
                    || msg.contains("timed out")
                    || msg.contains("ServiceUnavailable")
                    || msg.contains("503")
            }
            _ => false,
        }
    }

    /// 判断是否为服务端错误（非客户端责任）
    ///
    /// 服务端错误包括：
    /// - 内部错误、资源耗尽
    /// - 模型/计算错误
    /// - 服务端配置问题
    ///
    /// # 注意
    ///
    /// IO 错误和 HTTP 错误需要根据具体信息判断。
    pub fn is_server_error(&self) -> bool {
        match self {
            LidarAiError::ToolExecution { code, .. } => code.is_server_error(),
            // IO 错误中的资源问题可能是服务端错误
            LidarAiError::Io(e) => {
                matches!(
                    e.kind(),
                    std::io::ErrorKind::OutOfMemory
                        | std::io::ErrorKind::Other
                )
            }
            // HTTP 5xx 错误是服务端错误
            LidarAiError::Http(msg) => {
                msg.contains("500")
                    || msg.contains("502")
                    || msg.contains("503")
                    || msg.contains("504")
                    || msg.contains("Internal Server Error")
            }
            _ => false,
        }
    }

    /// 创建工具执行错误
    pub fn tool_execution(code: ErrorCode, message: impl Into<String>) -> Self {
        LidarAiError::ToolExecution {
            code,
            message: message.into(),
        }
    }

    /// 创建 IO 错误
    pub fn io(err: std::io::Error) -> Self {
        LidarAiError::Io(err)
    }

    /// 创建 HTTP 错误
    pub fn http(msg: impl Into<String>) -> Self {
        LidarAiError::Http(msg.into())
    }
}

impl From<reqwest::Error> for LidarAiError {
    fn from(err: reqwest::Error) -> Self {
        // 更详细的错误分类
        if err.is_timeout() {
            LidarAiError::Http(format!("HTTP 请求超时：{}", err))
        } else if err.is_request() {
            LidarAiError::Http(format!("HTTP 请求错误：{}", err))
        } else if err.is_connect() {
            LidarAiError::Http(format!("HTTP 连接错误：{}", err))
        } else if err.is_body() {
            LidarAiError::Http(format!("HTTP 响应体错误：{}", err))
        } else if err.is_decode() {
            LidarAiError::Http(format!("HTTP 响应解码错误：{}", err))
        } else {
            LidarAiError::Http(format!("HTTP 错误：{}", err))
        }
    }
}

/// 结果类型别名
pub type Result<T> = std::result::Result<T, LidarAiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    /// 测试 LidarAiError 的 Display 实现
    #[test]
    fn test_lidar_ai_error_display() {
        let io_error = LidarAiError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "文件不存在",
        ));
        assert!(io_error.to_string().contains("IO 错误"));
        assert!(io_error.to_string().contains("文件不存在"));

        let json_error = LidarAiError::Json(serde_json::from_str::<Value>("invalid").unwrap_err());
        assert!(json_error.to_string().contains("JSON 解析错误"));

        let ipc_error = LidarAiError::IpcCommunication("连接断开".to_string());
        assert!(ipc_error.to_string().contains("IPC 通信错误"));
        assert!(ipc_error.to_string().contains("连接断开"));

        let tool_error = LidarAiError::ToolExecution {
            code: ErrorCode::FileNotFound,
            message: "测试文件不存在".to_string(),
        };
        assert!(tool_error.to_string().contains("工具执行错误"));

        let http_error = LidarAiError::Http("500 Internal Server Error".to_string());
        assert!(http_error.to_string().contains("HTTP 请求错误"));
        assert!(http_error.to_string().contains("500"));

        let config_error = LidarAiError::Config("配置无效".to_string());
        assert!(config_error.to_string().contains("配置错误"));
    }

    /// 测试 from_ipc_error 转换
    #[test]
    fn test_from_ipc_error() {
        let ipc_error = IpcError {
            code: ErrorCode::InvalidParameter,
            message: "参数验证失败".to_string(),
            details: Some(json!({"field": "name"})),
        };

        let lidar_error = LidarAiError::from_ipc_error(ipc_error);

        match &lidar_error {
            LidarAiError::ToolExecution { code, message } => {
                assert_eq!(*code, ErrorCode::InvalidParameter);
                assert_eq!(*message, "参数验证失败");
            }
            _ => panic!("Expected ToolExecution error"),
        }
    }

    /// 测试 error_code 方法
    #[test]
    fn test_error_code() {
        // ToolExecution 错误应该返回 Some
        let tool_error = LidarAiError::ToolExecution {
            code: ErrorCode::ModelNotLoaded,
            message: "模型未加载".to_string(),
        };
        assert_eq!(tool_error.error_code(), Some(&ErrorCode::ModelNotLoaded));

        // 其他错误应该返回 None
        let io_error = LidarAiError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "文件不存在",
        ));
        assert_eq!(io_error.error_code(), None);

        let http_error = LidarAiError::Http("请求失败".to_string());
        assert_eq!(http_error.error_code(), None);
    }

    /// 测试 is_recoverable 方法
    #[test]
    fn test_is_recoverable() {
        // 可恢复错误
        let recoverable_error = LidarAiError::ToolExecution {
            code: ErrorCode::CommunicationTimeout,
            message: "通信超时".to_string(),
        };
        assert!(recoverable_error.is_recoverable());

        // 不可恢复错误
        let non_recoverable_error = LidarAiError::ToolExecution {
            code: ErrorCode::FileNotFound,
            message: "文件不存在".to_string(),
        };
        assert!(!non_recoverable_error.is_recoverable());

        // IO 超时错误可恢复
        let io_timeout = LidarAiError::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "连接超时",
        ));
        assert!(io_timeout.is_recoverable());

        // IO 其他错误不可恢复
        let io_not_found = LidarAiError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "文件不存在",
        ));
        assert!(!io_not_found.is_recoverable());

        // HTTP 超时错误可恢复
        let http_timeout = LidarAiError::Http("request timed out".to_string());
        assert!(http_timeout.is_recoverable());

        // HTTP 其他错误不可恢复
        let http_not_found = LidarAiError::Http("404 Not Found".to_string());
        assert!(!http_not_found.is_recoverable());
    }

    /// 测试 is_server_error 方法
    #[test]
    fn test_is_server_error() {
        // 服务端错误
        let server_error = LidarAiError::ToolExecution {
            code: ErrorCode::InternalError,
            message: "内部错误".to_string(),
        };
        assert!(server_error.is_server_error());

        // 客户端错误
        let client_error = LidarAiError::ToolExecution {
            code: ErrorCode::InvalidParameter,
            message: "参数错误".to_string(),
        };
        assert!(!client_error.is_server_error());

        // HTTP 5xx 错误是服务端错误
        let http_500 = LidarAiError::Http("500 Internal Server Error".to_string());
        assert!(http_500.is_server_error());

        // HTTP 4xx 错误不是服务端错误
        let http_404 = LidarAiError::Http("404 Not Found".to_string());
        assert!(!http_404.is_server_error());
    }

    /// 测试 tool_execution 构造方法
    #[test]
    fn test_tool_execution_constructor() {
        let error = LidarAiError::tool_execution(ErrorCode::ModelNotLoaded, "模型未加载");
        match error {
            LidarAiError::ToolExecution { code, message } => {
                assert_eq!(code, ErrorCode::ModelNotLoaded);
                assert_eq!(message, "模型未加载");
            }
            _ => panic!("Expected ToolExecution"),
        }
    }

    /// 测试 http 构造方法
    #[test]
    fn test_http_constructor() {
        let error = LidarAiError::http("连接失败");
        match error {
            LidarAiError::Http(msg) => {
                assert_eq!(msg, "连接失败");
            }
            _ => panic!("Expected Http"),
        }
    }

    /// 测试 io 构造方法
    #[test]
    fn test_io_constructor() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "文件不存在");
        let error = LidarAiError::io(io_err);
        match error {
            LidarAiError::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            }
            _ => panic!("Expected Io"),
        }
    }

    /// 测试所有错误类型的 Display
    #[test]
    fn test_all_error_variants_display() {
        let errors = vec![
            LidarAiError::Io(std::io::Error::new(std::io::ErrorKind::Other, "io")),
            LidarAiError::Json(serde_json::from_str::<Value>("x").unwrap_err()),
            LidarAiError::IpcCommunication("ipc".to_string()),
            LidarAiError::ToolExecution {
                code: ErrorCode::InternalError,
                message: "tool".to_string(),
            },
            LidarAiError::AiScheduler("ai".to_string()),
            LidarAiError::Http("http".to_string()),
            LidarAiError::Config("config".to_string()),
        ];

        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }
}
