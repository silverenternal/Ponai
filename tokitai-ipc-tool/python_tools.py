"""
Python 工具服务端 - 通过 stdin/stdout JSON Lines 与 Rust 通信
"""
import sys
import json
import hashlib
import subprocess
from typing import Any


def sha256_hash(text: str) -> dict:
    """计算字符串的 SHA256 哈希值"""
    result = hashlib.sha256(text.encode()).hexdigest()
    return {"result": result}


def python_version() -> dict:
    """获取 Python 版本信息"""
    return {"version": sys.version, "platform": sys.platform}


def run_command(cmd: str, timeout: int = 30) -> dict:
    """执行 shell 命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": str(e)}


def calculate(a: float, b: float, operation: str) -> dict:
    """执行数学计算"""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf'),
        "power": lambda x, y: x ** y,
    }
    
    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}
    
    try:
        result = operations[operation](a, b)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# 工具注册表
TOOLS = {
    "sha256_hash": {
        "func": sha256_hash,
        "description": "计算字符串的 SHA256 哈希值",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要哈希的字符串"}
            },
            "required": ["text"]
        }
    },
    "python_version": {
        "func": python_version,
        "description": "获取 Python 版本和平台信息",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "run_command": {
        "func": run_command,
        "description": "执行 shell 命令并返回输出",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "要执行的命令"},
                "timeout": {"type": "integer", "description": "超时时间（秒）", "default": 30}
            },
            "required": ["cmd"]
        }
    },
    "calculate": {
        "func": calculate,
        "description": "执行数学计算（加减乘除幂）",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个操作数"},
                "b": {"type": "number", "description": "第二个操作数"},
                "operation": {
                    "type": "string", 
                    "description": "操作类型",
                    "enum": ["add", "subtract", "multiply", "divide", "power"]
                }
            },
            "required": ["a", "b", "operation"]
        }
    }
}


def handle_request(request: dict) -> dict:
    """处理单个请求"""
    tool_name = request.get("tool")
    args = request.get("args", {})
    
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}
    
    try:
        func = TOOLS[tool_name]["func"]
        result = func(**args)
        return {"tool": tool_name, "result": result}
    except Exception as e:
        return {"tool": tool_name, "error": str(e)}


def main():
    """主循环：从 stdin 读取 JSON 请求，输出 JSON 响应"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as e:
            response = {"error": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            response = {"error": f"Unexpected error: {str(e)}"}
        
        # 输出 JSON 响应
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
