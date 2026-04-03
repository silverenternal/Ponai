#!/usr/bin/env python3
"""
实例分割 HTTP 服务 - FastAPI 实现
支持 GPU 加速推理，可独立部署

用法:
    python instance_seg_server.py --host 0.0.0.0 --port 8080 --api-key your-secret-key
"""
import os
import sys
import time
import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 导入实例分割工具
from instance_seg_tools import (
    load_model, run_segmentation, get_result,
    visualize, export_result, _TEMP_PCD_PATH,
    _model, _result
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ 全局状态 ============
_model: Optional[Any] = None
_result: Optional[Dict[str, Any]] = None

# ============ API 密钥验证 ============
API_KEY = os.getenv("INSTANCE_SEG_API_KEY", None)

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """验证 API Key（如果配置了的话）"""
    if API_KEY is None:
        return True  # 未配置则跳过验证
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ============ 请求/响应模型 ============

class ToolRequest(BaseModel):
    """工具调用请求"""
    args: Dict[str, Any] = Field(default_factory=dict, description="工具参数")

class ToolResponse(BaseModel):
    """工具调用响应"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    backend: str
    version: str

class LoadModelRequest(BaseModel):
    """加载模型请求"""
    model_path: str
    model_type: str = "onnx"
    device: str = "cpu"

class BatchSegmentationRequest(BaseModel):
    """批量分割请求"""
    pointclouds: list = Field(description="点云数据列表，每个点云为 [[x,y,z], ...] 格式")
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.3

class BatchSegmentationResponse(BaseModel):
    """批量分割响应"""
    results: list
    total_time_ms: float
    count: int

# ============ 生命周期管理 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("实例分割服务启动中...")
    logger.info(f"API Key 验证：{'已启用' if API_KEY else '未启用'}")
    yield
    logger.info("实例分割服务关闭")

# ============ 创建 FastAPI 应用 ============

app = FastAPI(
    title="Lidar AI Studio - Instance Segmentation Service",
    description="3D 点云实例分割 HTTP 服务 - 支持 ONNX Runtime / PyTorch 后端",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件（允许跨域调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ API 端点 ============

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    from instance_seg_tools import ONNX_SUPPORT, TORCH_SUPPORT
    
    return HealthResponse(
        status="healthy",
        model_loaded=(_model is not None),
        backend="onnx" if ONNX_SUPPORT else ("pytorch" if TORCH_SUPPORT else "none"),
        version="1.0.0"
    )

@app.get("/api/v1/get_backend_info", response_model=ToolResponse, tags=["工具"])
async def get_backend_info(verified: bool = Depends(verify_api_key)):
    """获取后端信息"""
    from instance_seg_tools import ONNX_SUPPORT, TORCH_SUPPORT
    
    return ToolResponse(
        result={
            "backend": "onnx" if ONNX_SUPPORT else ("pytorch" if TORCH_SUPPORT else "none"),
            "onnx_support": ONNX_SUPPORT,
            "torch_support": TORCH_SUPPORT,
            "model_loaded": _model is not None,
        },
        error=None
    )

@app.post("/api/v1/load_instance_segmentation_model", response_model=ToolResponse, tags=["工具"])
async def api_load_model(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """加载实例分割模型"""
    global _model
    args = request.args
    try:
        result = load_model(
            model_path=args.get("model_path"),
            model_type=args.get("model_type", "onnx"),
            device=args.get("device", "cpu")
        )
        
        # 更新全局状态
        if "error" not in result:
            _model = result.get("model")
        
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"加载模型失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/run_instance_segmentation", response_model=ToolResponse, tags=["工具"])
async def api_run_segmentation(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """执行实例分割"""
    global _result
    args = request.args
    try:
        result = run_segmentation(
            confidence_threshold=args.get("confidence_threshold", 0.5),
            iou_threshold=args.get("iou_threshold", 0.3)
        )
        
        # 更新全局状态
        if "error" not in result:
            _result = result.get("data")
        
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/get_segmentation_result", response_model=ToolResponse, tags=["工具"])
async def api_get_result(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """获取分割结果"""
    try:
        result = get_result()
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"获取结果失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/visualize_segmentation", response_model=ToolResponse, tags=["工具"])
async def api_visualize(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """可视化分割结果"""
    try:
        result = visualize()
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"可视化失败：{e}")
        return ToolResponse(result=None, error=str(e))

@app.post("/api/v1/export_segmentation", response_model=ToolResponse, tags=["工具"])
async def api_export(
    request: ToolRequest,
    verified: bool = Depends(verify_api_key)
):
    """导出分割结果"""
    args = request.args
    try:
        result = export_result(
            output_path=args.get("output_path"),
            format=args.get("format", "json")
        )
        return ToolResponse(result=result, error=None)
    except Exception as e:
        logger.error(f"导出失败：{e}")
        return ToolResponse(result=None, error=str(e))

# ============ 批处理端点（高性能场景） ============

@app.post("/api/v1/batch_segmentation", response_model=BatchSegmentationResponse, tags=["批处理"])
async def api_batch_segmentation(
    request: BatchSegmentationRequest,
    verified: bool = Depends(verify_api_key)
):
    """批量实例分割（高性能场景）"""
    start_time = time.time()
    results = []
    
    for points in request.pointclouds:
        try:
            # 保存临时点云
            np.save(_TEMP_PCD_PATH, np.array(points))
            # 执行分割
            run_segmentation(
                confidence_threshold=request.confidence_threshold,
                iou_threshold=request.iou_threshold
            )
            # 获取结果
            seg_result = get_result()
            results.append(seg_result)
        except Exception as e:
            logger.error(f"批量处理中单个点云分割失败：{e}")
            results.append({"error": str(e)})
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchSegmentationResponse(
        results=results,
        total_time_ms=total_time,
        count=len(results)
    )

# ============ 错误处理 ============

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常：{exc}", exc_info=True)
    return ToolResponse(
        result=None,
        error=f"服务器内部错误：{str(exc)}"
    )

# ============ 主程序入口 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实例分割 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--api-key", type=str, default=None, help="API Key")
    parser.add_argument("--reload", action="store_true", help="开发模式：自动重载")
    
    args = parser.parse_args()
    
    if args.api_key:
        os.environ["INSTANCE_SEG_API_KEY"] = args.api_key
        API_KEY = args.api_key
    
    logger.info(f"启动服务：http://{args.host}:{args.port}")
    logger.info(f"工作进程数：{args.workers}")
    
    uvicorn.run(
        "instance_seg_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info"
    )
