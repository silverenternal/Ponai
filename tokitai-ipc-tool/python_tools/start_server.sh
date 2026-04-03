#!/bin/bash
# 实例分割 HTTP 服务启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
WORKERS="${WORKERS:-1}"
API_KEY="${API_KEY:-}"
RELOAD="${RELOAD:-false}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --reload)
            RELOAD="true"
            shift
            ;;
        --help)
            echo "用法：$0 [选项]"
            echo ""
            echo "选项:"
            echo "  --host HOST       监听地址 (默认：0.0.0.0)"
            echo "  --port PORT       监听端口 (默认：8080)"
            echo "  --workers N       工作进程数 (默认：1)"
            echo "  --api-key KEY     API Key (可选)"
            echo "  --reload          开发模式：自动重载"
            echo "  --help            显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项：$1"
            exit 1
            ;;
    esac
done

# 构建 uvicorn 命令
CMD="python instance_seg_server.py --host $HOST --port $PORT"

if [ "$WORKERS" -gt 1 ]; then
    CMD="$CMD --workers $WORKERS"
fi

if [ "$RELOAD" = "true" ]; then
    CMD="$CMD --reload"
fi

if [ -n "$API_KEY" ]; then
    CMD="$CMD --api-key $API_KEY"
    export INSTANCE_SEG_API_KEY="$API_KEY"
fi

echo "==================================="
echo "实例分割 HTTP 服务"
echo "==================================="
echo "监听地址：http://$HOST:$PORT"
echo "工作进程：$WORKERS"
echo "API Key:   ${API_KEY:-未启用}"
echo "重载模式：$RELOAD"
echo "==================================="
echo ""

# 启动服务
exec $CMD
