//! 共享 IPC 类型定义
//!
//! 包含 IPC 通信和 HTTP 后端共享的请求/响应结构

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ipc_error::IpcError;

/// IPC 请求结构
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IpcRequest {
    /// 工具名称
    pub tool: String,
    /// 工具参数（可选，默认为空对象）
    #[serde(default)]
    pub args: Value,
}

/// IPC 响应结构（使用结构化错误）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IpcResponse {
    /// 执行结果（成功时）
    pub result: Option<Value>,
    /// 错误信息（失败时）
    pub error: Option<IpcError>,
}

/// API 响应结构（HTTP 模式）
/// 与 IpcResponse 保持兼容，确保双后端统一的响应格式
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiResponse {
    /// 执行结果（成功时）
    pub result: Option<Value>,
    /// 错误信息（失败时）
    pub error: Option<IpcError>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_error::ErrorCode;
    use serde_json::json;

    /// 测试 IpcRequest 序列化
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

    /// 测试 IpcRequest 默认 args
    #[test]
    fn test_ipc_request_default_args() {
        let json_str = r#"{"tool": "test_tool"}"#;
        let request: IpcRequest = serde_json::from_str(json_str).unwrap();

        assert_eq!(request.tool, "test_tool");
        assert!(request.args.is_null());
    }

    /// 测试 IpcResponse 成功场景
    #[test]
    fn test_ipc_response_success() {
        let response = IpcResponse {
            result: Some(json!({"status": "ok", "data": [1, 2, 3]})),
            error: None,
        };

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: IpcResponse = serde_json::from_str(&serialized).unwrap();

        assert_eq!(response.result, deserialized.result);
        assert_eq!(response.error, deserialized.error);
    }

    /// 测试 IpcResponse 错误场景
    #[test]
    fn test_ipc_response_error() {
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

        assert!(deserialized.result.is_none());
        assert!(deserialized.error.is_some());
        assert_eq!(deserialized.error.unwrap().code, ErrorCode::FileNotFound);
    }

    /// 测试 ApiResponse 与 IpcResponse 兼容性
    #[test]
    fn test_api_response_compatibility() {
        // 创建一个 IpcResponse
        let ipc_response = IpcResponse {
            result: Some(json!({"data": "test"})),
            error: None,
        };

        // 序列化为 JSON
        let json_str = serde_json::to_string(&ipc_response).unwrap();

        // 反序列化为 ApiResponse
        let api_response: ApiResponse = serde_json::from_str(&json_str).unwrap();

        assert_eq!(ipc_response.result, api_response.result);
        assert_eq!(ipc_response.error, api_response.error);
    }

    /// 测试 IpcRequest Clone
    #[test]
    fn test_ipc_request_clone() {
        let request = IpcRequest {
            tool: "test".to_string(),
            args: json!({"key": "value"}),
        };

        let cloned = request.clone();
        assert_eq!(request.tool, cloned.tool);
        assert_eq!(request.args, cloned.args);
    }

    /// 测试 IpcResponse Clone
    #[test]
    fn test_ipc_response_clone() {
        let response = IpcResponse {
            result: Some(json!({"status": "ok"})),
            error: None,
        };

        let cloned = response.clone();
        assert_eq!(response.result, cloned.result);
        assert_eq!(response.error, cloned.error);
    }
}
