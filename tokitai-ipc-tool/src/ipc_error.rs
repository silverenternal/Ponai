//! 统一的 IPC 错误类型定义
//!
//! 支持跨语言 (Rust/Python/C++) 的结构化错误通信

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;

/// IPC 错误码 - 跨语言统一的错误分类
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    // ==================== 通用错误 ====================
    /// 无效的请求格式
    InvalidRequest,
    /// 工具不存在
    ToolNotFound,
    /// 内部错误
    InternalError,
    /// 未实现的功能
    NotImplemented,

    // ==================== 文件相关错误 ====================
    /// 文件不存在
    FileNotFound,
    /// 文件格式不支持
    FileFormatUnsupported,
    /// 文件读取失败
    FileReadError,
    /// 文件写入失败
    FileWriteError,
    /// 权限不足
    PermissionDenied,

    // ==================== 参数相关错误 ====================
    /// 无效的参数
    InvalidParameter,
    /// 缺少必填参数
    MissingRequiredParameter,
    /// 参数超出范围
    ParameterOutOfRange,
    /// 参数类型错误
    ParameterTypeMismatch,

    // ==================== 模型相关错误 ====================
    /// 模型未加载
    ModelNotLoaded,
    /// 模型加载失败
    ModelLoadFailed,
    /// 模型推理失败
    ModelInferenceFailed,
    /// 不支持的模型格式
    ModelFormatUnsupported,

    // ==================== 资源相关错误 ====================
    /// 内存不足
    OutOfMemory,
    /// 设备不可用
    DeviceNotAvailable,
    /// 资源已耗尽
    ResourceExhausted,

    // ==================== IPC 通信错误 ====================
    /// 进程已退出
    ProcessExited,
    /// 通信超时
    CommunicationTimeout,
    /// 连接断开
    ConnectionLost,

    // ==================== HTTP 特定错误 ====================
    /// HTTP 状态码错误（附带状态码）
    #[serde(rename = "http_status")]
    HttpStatus(u16),
    /// 认证失败
    AuthenticationFailed,
    /// 授权失败（权限不足）
    AuthorizationFailed,
    /// 服务不可用
    ServiceUnavailable,
    /// 请求频率超限
    RateLimitExceeded,
    /// 网关错误
    GatewayError,

    // ==================== 数据验证错误 ====================
    /// 数据验证失败
    ValidationFailed,
    /// 数据结构不匹配
    SchemaMismatch,
    /// 约束违反
    ConstraintViolation,
    /// 数据格式错误
    DataFormatError,

    // ==================== 并发/状态错误 ====================
    /// 互斥锁中毒（Mutex 被 panic 污染）
    ///
    /// 当持有锁的线程 panic 时，Mutex 会进入"中毒"状态，
    /// 后续锁获取会返回 `PoisonError`。
    LockPoisoned,
    /// 竞态条件检测到
    ///
    /// # 实现说明
    ///
    /// 此错误码为**保留错误码**，用于未来实现竞态检测机制。
    /// 当前版本不会主动返回此错误，但预留用于跨语言通信。
    RaceConditionDetected,
    /// 死锁检测到
    ///
    /// # 实现说明
    ///
    /// 此错误码为**保留错误码**。Rust 标准库的 `Mutex` 无法检测死锁，
    /// 此错误码用于跨语言通信场景（如 Python/C++ 端检测到死锁后传递过来）。
    /// 当前 Rust 端不会主动返回此错误。
    DeadlockDetected,
    /// 状态冲突
    ///
    /// 操作与当前系统状态冲突（如：在已关闭的连接上发送数据）。
    StateConflict,

    // ==================== 自定义错误 ====================
    /// 自定义错误码（用于扩展）
    #[serde(rename = "custom")]
    Custom(String),
}

impl ErrorCode {
    /// 判断是否为可恢复错误
    ///
    /// 可恢复错误包括：
    /// - 通信超时、连接问题（可重试）
    /// - 服务暂时不可用（可切换到备用服务）
    /// - 请求频率超限（等待后可重试）
    ///
    /// # 注意
    ///
    /// 以下错误**不**视为可恢复：
    /// - `OutOfMemory` / `ResourceExhausted`：资源耗尽可能需要外部干预
    /// - `LockPoisoned`：锁中毒表明代码有 bug，重试无法解决
    /// - `DeadlockDetected`：死锁需要重新设计锁顺序
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            // 通信相关 - 可重试
            ErrorCode::CommunicationTimeout
                | ErrorCode::ConnectionLost
                | ErrorCode::ProcessExited
                // HTTP 相关 - 部分可重试
                | ErrorCode::ServiceUnavailable
                | ErrorCode::GatewayError
                | ErrorCode::RateLimitExceeded
                // 状态冲突 - 可能暂时性
                | ErrorCode::StateConflict
        )
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
    /// `LockPoisoned` 视为服务端错误（代码 bug），
    /// `DeadlockDetected` 视为服务端错误（设计问题）。
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            // 内部错误
            ErrorCode::InternalError
                | ErrorCode::NotImplemented
                // 模型相关
                | ErrorCode::ModelLoadFailed
                | ErrorCode::ModelInferenceFailed
                // 资源相关
                | ErrorCode::OutOfMemory
                | ErrorCode::DeviceNotAvailable
                | ErrorCode::ResourceExhausted
                // HTTP 服务端错误
                | ErrorCode::ServiceUnavailable
                | ErrorCode::GatewayError
                // 并发/状态错误
                | ErrorCode::LockPoisoned
                | ErrorCode::DeadlockDetected
                | ErrorCode::RaceConditionDetected
                // 数据/计算错误
                | ErrorCode::ModelFormatUnsupported
                | ErrorCode::FileFormatUnsupported
        )
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::InvalidRequest => write!(f, "INVALID_REQUEST"),
            ErrorCode::ToolNotFound => write!(f, "TOOL_NOT_FOUND"),
            ErrorCode::InternalError => write!(f, "INTERNAL_ERROR"),
            ErrorCode::NotImplemented => write!(f, "NOT_IMPLEMENTED"),
            ErrorCode::FileNotFound => write!(f, "FILE_NOT_FOUND"),
            ErrorCode::FileFormatUnsupported => write!(f, "FILE_FORMAT_UNSUPPORTED"),
            ErrorCode::FileReadError => write!(f, "FILE_READ_ERROR"),
            ErrorCode::FileWriteError => write!(f, "FILE_WRITE_ERROR"),
            ErrorCode::PermissionDenied => write!(f, "PERMISSION_DENIED"),
            ErrorCode::InvalidParameter => write!(f, "INVALID_PARAMETER"),
            ErrorCode::MissingRequiredParameter => write!(f, "MISSING_REQUIRED_PARAMETER"),
            ErrorCode::ParameterOutOfRange => write!(f, "PARAMETER_OUT_OF_RANGE"),
            ErrorCode::ParameterTypeMismatch => write!(f, "PARAMETER_TYPE_MISMATCH"),
            ErrorCode::ModelNotLoaded => write!(f, "MODEL_NOT_LOADED"),
            ErrorCode::ModelLoadFailed => write!(f, "MODEL_LOAD_FAILED"),
            ErrorCode::ModelInferenceFailed => write!(f, "MODEL_INFERENCE_FAILED"),
            ErrorCode::ModelFormatUnsupported => write!(f, "MODEL_FORMAT_UNSUPPORTED"),
            ErrorCode::OutOfMemory => write!(f, "OUT_OF_MEMORY"),
            ErrorCode::DeviceNotAvailable => write!(f, "DEVICE_NOT_AVAILABLE"),
            ErrorCode::ResourceExhausted => write!(f, "RESOURCE_EXHAUSTED"),
            ErrorCode::ProcessExited => write!(f, "PROCESS_EXITED"),
            ErrorCode::CommunicationTimeout => write!(f, "COMMUNICATION_TIMEOUT"),
            ErrorCode::ConnectionLost => write!(f, "CONNECTION_LOST"),
            // HTTP 特定错误
            ErrorCode::HttpStatus(code) => write!(f, "HTTP_{}", code),
            ErrorCode::AuthenticationFailed => write!(f, "AUTHENTICATION_FAILED"),
            ErrorCode::AuthorizationFailed => write!(f, "AUTHORIZATION_FAILED"),
            ErrorCode::ServiceUnavailable => write!(f, "SERVICE_UNAVAILABLE"),
            ErrorCode::RateLimitExceeded => write!(f, "RATE_LIMIT_EXCEEDED"),
            ErrorCode::GatewayError => write!(f, "GATEWAY_ERROR"),
            // 数据验证错误
            ErrorCode::ValidationFailed => write!(f, "VALIDATION_FAILED"),
            ErrorCode::SchemaMismatch => write!(f, "SCHEMA_MISMATCH"),
            ErrorCode::ConstraintViolation => write!(f, "CONSTRAINT_VIOLATION"),
            ErrorCode::DataFormatError => write!(f, "DATA_FORMAT_ERROR"),
            // 并发/状态错误
            ErrorCode::LockPoisoned => write!(f, "LOCK_POISONED"),
            ErrorCode::RaceConditionDetected => write!(f, "RACE_CONDITION_DETECTED"),
            ErrorCode::DeadlockDetected => write!(f, "DEADLOCK_DETECTED"),
            ErrorCode::StateConflict => write!(f, "STATE_CONFLICT"),
            // 自定义错误
            ErrorCode::Custom(code) => write!(f, "CUSTOM:{}", code),
        }
    }
}

impl From<&str> for ErrorCode {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "invalid_request" => ErrorCode::InvalidRequest,
            "tool_not_found" => ErrorCode::ToolNotFound,
            "internal_error" => ErrorCode::InternalError,
            "not_implemented" => ErrorCode::NotImplemented,
            "file_not_found" => ErrorCode::FileNotFound,
            "file_format_unsupported" => ErrorCode::FileFormatUnsupported,
            "file_read_error" => ErrorCode::FileReadError,
            "file_write_error" => ErrorCode::FileWriteError,
            "permission_denied" => ErrorCode::PermissionDenied,
            "invalid_parameter" => ErrorCode::InvalidParameter,
            "missing_required_parameter" => ErrorCode::MissingRequiredParameter,
            "parameter_out_of_range" => ErrorCode::ParameterOutOfRange,
            "parameter_type_mismatch" => ErrorCode::ParameterTypeMismatch,
            "model_not_loaded" => ErrorCode::ModelNotLoaded,
            "model_load_failed" => ErrorCode::ModelLoadFailed,
            "model_inference_failed" => ErrorCode::ModelInferenceFailed,
            "model_format_unsupported" => ErrorCode::ModelFormatUnsupported,
            "out_of_memory" => ErrorCode::OutOfMemory,
            "device_not_available" => ErrorCode::DeviceNotAvailable,
            "resource_exhausted" => ErrorCode::ResourceExhausted,
            "process_exited" => ErrorCode::ProcessExited,
            "communication_timeout" => ErrorCode::CommunicationTimeout,
            "connection_lost" => ErrorCode::ConnectionLost,
            // HTTP 特定错误
            "authentication_failed" => ErrorCode::AuthenticationFailed,
            "authorization_failed" => ErrorCode::AuthorizationFailed,
            "service_unavailable" => ErrorCode::ServiceUnavailable,
            "rate_limit_exceeded" => ErrorCode::RateLimitExceeded,
            "gateway_error" => ErrorCode::GatewayError,
            // 数据验证错误
            "validation_failed" => ErrorCode::ValidationFailed,
            "schema_mismatch" => ErrorCode::SchemaMismatch,
            "constraint_violation" => ErrorCode::ConstraintViolation,
            "data_format_error" => ErrorCode::DataFormatError,
            // 并发/状态错误
            "lock_poisoned" => ErrorCode::LockPoisoned,
            "race_condition_detected" => ErrorCode::RaceConditionDetected,
            "deadlock_detected" => ErrorCode::DeadlockDetected,
            "state_conflict" => ErrorCode::StateConflict,
            // 自定义错误（未知错误码）
            other => ErrorCode::Custom(other.to_string()),
        }
    }
}

/// IPC 结构化错误
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IpcError {
    /// 错误码
    pub code: ErrorCode,
    /// 错误消息
    pub message: String,
    /// 附加详情（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl IpcError {
    /// 创建新的 IPC 错误
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
        }
    }

    /// 创建带详情的 IPC 错误
    pub fn with_details(code: ErrorCode, message: impl Into<String>, details: Value) -> Self {
        Self {
            code,
            message: message.into(),
            details: Some(details),
        }
    }

    /// 创建文件不存在错误
    pub fn file_not_found(path: impl Into<String>) -> Self {
        let path_str = path.into();
        Self::with_details(
            ErrorCode::FileNotFound,
            format!("文件不存在：{}", path_str),
            json!({"path": path_str}),
        )
    }

    /// 创建文件格式不支持错误
    pub fn file_format_unsupported(format: impl Into<String>) -> Self {
        let fmt = format.into();
        Self::with_details(
            ErrorCode::FileFormatUnsupported,
            format!("不支持的文件格式：{}", fmt),
            json!({"format": fmt}),
        )
    }

    /// 创建参数错误
    pub fn invalid_parameter(param: impl Into<String>, reason: impl Into<String>) -> Self {
        let param_str = param.into();
        let reason_str = reason.into();
        Self::with_details(
            ErrorCode::InvalidParameter,
            format!("参数 '{}' 无效：{}", param_str, reason_str),
            json!({"parameter": param_str, "reason": reason_str}),
        )
    }

    /// 创建缺少参数错误
    pub fn missing_parameter(param: impl Into<String>) -> Self {
        let param_str = param.into();
        Self::with_details(
            ErrorCode::MissingRequiredParameter,
            format!("缺少必填参数：{}", param_str),
            json!({"parameter": param_str}),
        )
    }

    /// 创建模型未加载错误
    pub fn model_not_loaded() -> Self {
        Self::new(
            ErrorCode::ModelNotLoaded,
            "模型未加载，请先调用 load_model",
        )
    }

    /// 创建工具不存在错误
    pub fn tool_not_found(tool_name: impl Into<String>) -> Self {
        let name = tool_name.into();
        Self::with_details(
            ErrorCode::ToolNotFound,
            format!("未知工具：{}", name),
            json!({"tool": name}),
        )
    }

    /// 创建内部错误
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InternalError, message)
    }
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(details) = &self.details {
            write!(f, " (详情：{})", details)?;
        }
        Ok(())
    }
}

impl std::error::Error for IpcError {}

/// 从字符串解析 ErrorCode（用于接收 Python/C++ 的错误码）
pub fn parse_error_code(code_str: &str) -> ErrorCode {
    code_str.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_serialization() {
        let code = ErrorCode::FileNotFound;
        let serialized = serde_json::to_string(&code).unwrap();
        assert_eq!(serialized, "\"file_not_found\"");
    }

    #[test]
    fn test_error_code_deserialization() {
        let json = "\"invalid_parameter\"";
        let code: ErrorCode = serde_json::from_str(json).unwrap();
        assert_eq!(code, ErrorCode::InvalidParameter);
    }

    #[test]
    fn test_ipc_error_with_details() {
        let error = IpcError::file_not_found("/path/to/file.pcd");
        assert_eq!(error.code, ErrorCode::FileNotFound);
        assert!(error.message.contains("/path/to/file.pcd"));
        assert!(error.details.is_some());
    }

    #[test]
    fn test_is_recoverable() {
        // 可恢复错误 - 通信/HTTP 相关
        assert!(ErrorCode::CommunicationTimeout.is_recoverable());
        assert!(ErrorCode::ConnectionLost.is_recoverable());
        assert!(ErrorCode::ServiceUnavailable.is_recoverable());
        assert!(ErrorCode::RateLimitExceeded.is_recoverable());
        assert!(ErrorCode::GatewayError.is_recoverable());
        assert!(ErrorCode::StateConflict.is_recoverable());

        // 不可恢复错误 - 资源耗尽/代码 bug
        assert!(!ErrorCode::OutOfMemory.is_recoverable());  // 资源耗尽可能需要外部干预
        assert!(!ErrorCode::LockPoisoned.is_recoverable()); // 锁中毒表明代码有 bug
        assert!(!ErrorCode::DeadlockDetected.is_recoverable()); // 死锁需要重新设计

        // 不可恢复错误 - 客户端错误
        assert!(!ErrorCode::InvalidParameter.is_recoverable());
        assert!(!ErrorCode::FileNotFound.is_recoverable());
        assert!(!ErrorCode::InternalError.is_recoverable());
    }

    #[test]
    fn test_is_server_error() {
        // 服务端错误
        assert!(ErrorCode::InternalError.is_server_error());
        assert!(ErrorCode::OutOfMemory.is_server_error());
        assert!(ErrorCode::ServiceUnavailable.is_server_error());
        assert!(ErrorCode::ModelLoadFailed.is_server_error());
        
        // 客户端错误
        assert!(!ErrorCode::InvalidParameter.is_server_error());
        assert!(!ErrorCode::FileNotFound.is_server_error());
    }
}
