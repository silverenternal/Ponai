"""
IPC 错误类型定义 - Python 端

与 Rust 端 ErrorCode 枚举保持一致，支持结构化错误返回
"""
from enum import Enum
from typing import Any, Dict, Optional, Union
import json


class ErrorCode(str, Enum):
    # ==================== 通用错误 ====================
    INVALID_REQUEST = "invalid_request"
    TOOL_NOT_FOUND = "tool_not_found"
    INTERNAL_ERROR = "internal_error"
    NOT_IMPLEMENTED = "not_implemented"
    CUSTOM = "custom"  # 自定义错误码

    # ==================== 文件相关错误 ====================
    FILE_NOT_FOUND = "file_not_found"
    FILE_FORMAT_UNSUPPORTED = "file_format_unsupported"
    FILE_READ_ERROR = "file_read_error"
    FILE_WRITE_ERROR = "file_write_error"
    PERMISSION_DENIED = "permission_denied"

    # ==================== 参数相关错误 ====================
    INVALID_PARAMETER = "invalid_parameter"
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    PARAMETER_OUT_OF_RANGE = "parameter_out_of_range"
    PARAMETER_TYPE_MISMATCH = "parameter_type_mismatch"

    # ==================== 模型相关错误 ====================
    MODEL_NOT_LOADED = "model_not_loaded"
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_INFERENCE_FAILED = "model_inference_failed"
    MODEL_FORMAT_UNSUPPORTED = "model_format_unsupported"

    # ==================== 资源相关错误 ====================
    OUT_OF_MEMORY = "out_of_memory"
    DEVICE_NOT_AVAILABLE = "device_not_available"
    RESOURCE_EXHAUSTED = "resource_exhausted"

    # ==================== IPC 通信错误 ====================
    PROCESS_EXITED = "process_exited"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    CONNECTION_LOST = "connection_lost"

    # ==================== HTTP 特定错误 ====================
    HTTP_STATUS = "http_status"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    GATEWAY_ERROR = "gateway_error"

    # ==================== 数据验证错误 ====================
    VALIDATION_FAILED = "validation_failed"
    SCHEMA_MISMATCH = "schema_mismatch"
    CONSTRAINT_VIOLATION = "constraint_violation"
    DATA_FORMAT_ERROR = "data_format_error"

    # ==================== 并发/状态错误 ====================
    LOCK_POISONED = "lock_poisoned"
    RACE_CONDITION_DETECTED = "race_condition_detected"
    DEADLOCK_DETECTED = "deadlock_detected"
    STATE_CONFLICT = "state_conflict"

    @classmethod
    def from_str(cls, code_str: str) -> "ErrorCode":
        """从字符串解析错误码
        
        Args:
            code_str: 错误码字符串（snake_case 格式）
            
        Returns:
            对应的 ErrorCode 枚举值，如果未知则返回 CUSTOM
        """
        try:
            return cls(code_str.lower())
        except ValueError:
            return cls.CUSTOM

    def is_recoverable(self) -> bool:
        """判断错误是否可恢复

        可恢复错误：
        - 通信超时、连接问题（可重试）
        - 资源暂时不可用（等待后可重试）
        - 服务暂时不可用（可切换到备用服务）
        """
        return self in (
            # 通信相关 - 可重试
            ErrorCode.COMMUNICATION_TIMEOUT,
            ErrorCode.CONNECTION_LOST,
            ErrorCode.PROCESS_EXITED,
            # HTTP 相关 - 部分可重试
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.GATEWAY_ERROR,
            ErrorCode.RATE_LIMIT_EXCEEDED,
            # 资源相关 - 等待后可重试
            ErrorCode.OUT_OF_MEMORY,
            ErrorCode.DEVICE_NOT_AVAILABLE,
            ErrorCode.RESOURCE_EXHAUSTED,
            # 并发相关 - 可重试
            ErrorCode.LOCK_POISONED,
            ErrorCode.RACE_CONDITION_DETECTED,
            ErrorCode.STATE_CONFLICT,
        )

    def is_server_error(self) -> bool:
        """判断是否为服务端错误（非客户端责任）

        服务端错误：
        - 内部错误、资源耗尽
        - 模型/计算错误
        - 服务端配置问题
        """
        return self in (
            # 内部错误
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.NOT_IMPLEMENTED,
            # 模型相关
            ErrorCode.MODEL_LOAD_FAILED,
            ErrorCode.MODEL_INFERENCE_FAILED,
            # 资源相关
            ErrorCode.OUT_OF_MEMORY,
            ErrorCode.DEVICE_NOT_AVAILABLE,
            ErrorCode.RESOURCE_EXHAUSTED,
            # HTTP 服务端错误
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.GATEWAY_ERROR,
            # 并发/状态错误
            ErrorCode.LOCK_POISONED,
            ErrorCode.DEADLOCK_DETECTED,
            # 数据/计算错误
            ErrorCode.MODEL_FORMAT_UNSUPPORTED,
            ErrorCode.FILE_FORMAT_UNSUPPORTED,
        )


class IpcError:
    """IPC 结构化错误"""

    def __init__(
        self,
        code: Union[ErrorCode, str],
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code.value if isinstance(code, ErrorCode) else code
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """字符串表示"""
        if self.details:
            return f"{self.code}: {self.message} (details: {self.details})"
        return f"{self.code}: {self.message}"

    def __repr__(self) -> str:
        """调试表示"""
        return f"IpcError(code={self.code!r}, message={self.message!r}, details={self.details!r})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        result = {
            "code": self.code,
            "message": self.message
        }
        if self.details is not None:
            result["details"] = self.details
        return result

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IpcError":
        """从字典创建 IpcError
        
        Args:
            data: 包含 code, message, details 的字典
            
        Returns:
            IpcError 实例
        """
        return cls(
            code=data.get("code", "internal_error"),
            message=data.get("message", "未知错误"),
            details=data.get("details")
        )

    @classmethod
    def from_json(cls, json_str: str) -> "IpcError":
        """从 JSON 字符串创建 IpcError
        
        Args:
            json_str: JSON 格式的错误字符串
            
        Returns:
            IpcError 实例
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def file_not_found(cls, path: str) -> "IpcError":
        """创建文件不存在错误"""
        return cls(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"文件不存在：{path}",
            details={"path": path}
        )

    @classmethod
    def file_format_unsupported(cls, file_format: str) -> "IpcError":
        """创建文件格式不支持错误"""
        return cls(
            code=ErrorCode.FILE_FORMAT_UNSUPPORTED,
            message=f"不支持的文件格式：{file_format}",
            details={"format": file_format}
        )

    @classmethod
    def invalid_parameter(cls, param: str, reason: str) -> "IpcError":
        """创建参数错误"""
        return cls(
            code=ErrorCode.INVALID_PARAMETER,
            message=f"参数 '{param}' 无效：{reason}",
            details={"parameter": param, "reason": reason}
        )

    @classmethod
    def missing_parameter(cls, param: str) -> "IpcError":
        """创建缺少参数错误"""
        return cls(
            code=ErrorCode.MISSING_REQUIRED_PARAMETER,
            message=f"缺少必填参数：{param}",
            details={"parameter": param}
        )

    @classmethod
    def parameter_out_of_range(cls, param: str, value: Any, min_val: Any = None, max_val: Any = None) -> "IpcError":
        """创建参数超出范围错误"""
        details = {"parameter": param, "value": value}
        if min_val is not None:
            details["min"] = min_val
        if max_val is not None:
            details["max"] = max_val
        return cls(
            code=ErrorCode.PARAMETER_OUT_OF_RANGE,
            message=f"参数 '{param}' 超出范围：{value}",
            details=details
        )

    @classmethod
    def model_not_loaded(cls) -> "IpcError":
        """创建模型未加载错误"""
        return cls(
            code=ErrorCode.MODEL_NOT_LOADED,
            message="模型未加载，请先调用 load_model"
        )

    @classmethod
    def tool_not_found(cls, tool_name: str) -> "IpcError":
        """创建工具不存在错误"""
        return cls(
            code=ErrorCode.TOOL_NOT_FOUND,
            message=f"未知工具：{tool_name}",
            details={"tool": tool_name}
        )

    @classmethod
    def internal_error(cls, message: str) -> "IpcError":
        """创建内部错误"""
        return cls(
            code=ErrorCode.INTERNAL_ERROR,
            message=message
        )


def create_error_response(error: IpcError) -> Dict[str, Any]:
    """创建错误响应字典"""
    return {
        "result": None,
        "error": error.to_dict()
    }


def create_success_response(result: Any) -> Dict[str, Any]:
    """创建成功响应字典"""
    return {
        "result": result,
        "error": None
    }


if __name__ == "__main__":
    # 简单的自测试
    import sys
    
    def test_error_code_from_str():
        assert ErrorCode.from_str("file_not_found") == ErrorCode.FILE_NOT_FOUND
        assert ErrorCode.from_str("FILE_NOT_FOUND") == ErrorCode.FILE_NOT_FOUND
        assert ErrorCode.from_str("unknown_error") == ErrorCode.CUSTOM
        print("✓ ErrorCode.from_str() 测试通过")
    
    def test_error_code_is_recoverable():
        assert ErrorCode.COMMUNICATION_TIMEOUT.is_recoverable() == True
        assert ErrorCode.CONNECTION_LOST.is_recoverable() == True
        assert ErrorCode.FILE_NOT_FOUND.is_recoverable() == False
        assert ErrorCode.INVALID_PARAMETER.is_recoverable() == False
        print("✓ ErrorCode.is_recoverable() 测试通过")
    
    def test_error_code_is_server_error():
        assert ErrorCode.INTERNAL_ERROR.is_server_error() == True
        assert ErrorCode.OUT_OF_MEMORY.is_server_error() == True
        assert ErrorCode.FILE_NOT_FOUND.is_server_error() == False
        assert ErrorCode.INVALID_PARAMETER.is_server_error() == False
        print("✓ ErrorCode.is_server_error() 测试通过")
    
    def test_ipc_error_str_repr():
        error = IpcError(
            code=ErrorCode.FILE_NOT_FOUND,
            message="文件不存在",
            details={"path": "/test"}
        )
        assert "file_not_found" in str(error)
        assert "文件不存在" in str(error)
        assert "IpcError" in repr(error)
        
        error_no_details = IpcError(
            code=ErrorCode.INTERNAL_ERROR,
            message="内部错误"
        )
        assert "internal_error" in str(error_no_details)
        assert "details" not in str(error_no_details)
        print("✓ IpcError.__str__ 和 __repr__ 测试通过")
    
    def test_ipc_error_from_dict():
        data = {
            "code": "invalid_parameter",
            "message": "参数错误",
            "details": {"field": "name"}
        }
        error = IpcError.from_dict(data)
        assert error.code == "invalid_parameter"
        assert error.message == "参数错误"
        assert error.details == {"field": "name"}
        print("✓ IpcError.from_dict() 测试通过")
    
    def test_ipc_error_from_json():
        json_str = '{"code": "model_not_loaded", "message": "模型未加载"}'
        error = IpcError.from_json(json_str)
        assert error.code == "model_not_loaded"
        assert error.message == "模型未加载"
        assert error.details is None
        print("✓ IpcError.from_json() 测试通过")
    
    def test_create_response():
        error = IpcError.internal_error("测试错误")
        error_response = create_error_response(error)
        assert error_response["result"] is None
        assert error_response["error"]["code"] == "internal_error"
        
        success_response = create_success_response({"data": [1, 2, 3]})
        assert success_response["result"] == {"data": [1, 2, 3]}
        assert success_response["error"] is None
        print("✓ create_error_response 和 create_success_response 测试通过")
    
    # 运行所有测试
    try:
        test_error_code_from_str()
        test_error_code_is_recoverable()
        test_error_code_is_server_error()
        test_ipc_error_str_repr()
        test_ipc_error_from_dict()
        test_ipc_error_from_json()
        test_create_response()
        print("\n✅ 所有 Python 端错误处理测试通过")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ 测试失败：{e}")
        sys.exit(1)
