#!/usr/bin/env python3
"""
PointPillars 3D 目标检测模型 - ONNX Runtime 推理
支持 KITTI 数据集的 3 类检测：Vehicle, Pedestrian, Cyclist

模型架构：Pillar 编码 + 2D CNN + 检测头
输入：点云 [1, 4096, 4] (x, y, z, intensity)
输出：边界框 [1, 7], 置信度 [1], 类别 [1]
"""
import sys
import json
import os
import time
import numpy as np
from typing import Any, Dict, Optional, List

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_SUPPORT = True
except ImportError:
    ONNX_SUPPORT = False

# 类别映射 (KITTI)
LABEL_MAP = {
    0: "Car",
    1: "Pedestrian", 
    2: "Cyclist",
}

# 全局状态
_model_session: Optional[ort.InferenceSession] = None
_model_path: Optional[str] = None


def load_pointpillars(model_path: str, device: str = "cpu") -> dict:
    """
    加载 PointPillars ONNX 模型
    
    :param model_path: ONNX 模型文件路径
    :param device: 推理设备 ("cpu" 或 "cuda")
    :return: 加载结果字典
    """
    global _model_session, _model_path
    
    if not os.path.exists(model_path):
        return {"error": f"模型文件不存在：{model_path}"}
    
    if not ONNX_SUPPORT:
        return {"error": "ONNX Runtime 未安装"}
    
    try:
        # 配置 ONNX Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # CPU 并行线程数
        
        # 选择执行提供者
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        _model_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        _model_path = model_path
        
        # 获取模型信息
        input_info = _model_session.get_inputs()[0]
        output_info = _model_session.get_outputs()
        
        return {
            "message": "✓ PointPillars 模型加载成功",
            "info": {
                "path": model_path,
                "type": "onnx",
                "device": device,
                "input_shape": input_info.shape,
                "input_name": input_info.name,
                "outputs": [{"name": o.name, "shape": o.shape} for o in output_info],
                "providers": _model_session.get_providers()
            }
        }
        
    except Exception as e:
        _model_session = None
        return {"error": f"模型加载失败：{str(e)}"}


def run_pointpillars(
    points: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5,
) -> dict:
    """
    执行 PointPillars 推理
    
    :param points: 点云数据 [N, 4] 或 [N, 3]
    :param confidence_threshold: 置信度阈值
    :return: 推理结果字典
    """
    global _model_session, _model_path
    
    if _model_session is None:
        return {"error": "模型未加载，请先调用 load_pointpillars"}
    
    # 加载点云文件
    if points is None:
        temp_pcd_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tmp",
            "lidar_ai_current_pcd.npy"
        )
        if not os.path.exists(temp_pcd_path):
            return {"error": f"点云文件不存在：{temp_pcd_path}"}
        points = np.load(temp_pcd_path)
    
    if len(points) == 0:
        return {"error": "点云数据为空"}
    
    try:
        start_time = time.time()
        
        # 预处理点云
        input_data = preprocess_points(points)
        
        # ONNX 推理
        outputs = _model_session.run(None, {'points': input_data})
        boxes, scores, labels = outputs
        
        # 后处理
        detections = postprocess_results(boxes, scores, labels, confidence_threshold)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "message": f"✓ PointPillars 推理完成，检测到 {len(detections)} 个目标",
            "summary": {
                "num_detections": len(detections),
                "inference_time_ms": round(inference_time, 2),
                "input_points": len(points),
                "categories": list(set(d["label"] for d in detections))
            },
            "detections": detections
        }
        
    except Exception as e:
        return {"error": f"推理失败：{str(e)}"}


def preprocess_points(points: np.ndarray, num_points: int = 4096) -> np.ndarray:
    """
    点云预处理：归一化、降采样、格式转换
    
    :param points: 原始点云 [N, 3] 或 [N, 4]
    :param num_points: 目标点数
    :return: 模型输入 [1, num_points, 4]
    """
    # 确保有 intensity 通道
    if points.shape[1] == 3:
        # 添加虚拟 intensity
        intensity = np.zeros((len(points), 1), dtype=np.float32)
        points = np.hstack([points, intensity])
    
    # 去中心化
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points.copy()
    points_centered[:, :3] -= centroid
    
    # 降采样 (如果点数过多)
    if len(points_centered) > num_points:
        indices = np.random.choice(len(points_centered), num_points, replace=False)
        points_centered = points_centered[indices]
    
    # 填充 (如果点数不足)
    if len(points_centered) < num_points:
        padding = np.zeros((num_points - len(points_centered), 4), dtype=np.float32)
        points_centered = np.vstack([points_centered, padding])
    
    # 添加 batch 维度 [1, N, 4]
    input_data = points_centered.astype(np.float32)[np.newaxis, :, :]
    
    return input_data


def postprocess_results(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float
) -> List[Dict[str, Any]]:
    """
    后处理：过滤、格式化检测结果
    
    :param boxes: 边界框 [1, 7] 或 [N, 7]
    :param scores: 置信度 [1] 或 [N]
    :param labels: 类别 [1] 或 [N]
    :param confidence_threshold: 置信度阈值
    :return: 检测结果列表
    """
    # 展平输出
    boxes = boxes.reshape(-1, 7)
    scores = scores.flatten()
    labels = labels.flatten()
    
    detections = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            label_id = int(labels[i])
            detections.append({
                "id": i,
                "label": LABEL_MAP.get(label_id, f"Unknown_{label_id}"),
                "label_id": label_id,
                "confidence": round(float(scores[i]), 4),
                "bbox_3d": {
                    "center": [float(x) for x in boxes[i, :3].round(4)],
                    "size": [float(x) for x in boxes[i, 3:6].round(4)],
                    "rotation": float(boxes[i, 6])
                }
            })
    
    # 按置信度排序
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    return detections


def get_pointpillars_info() -> dict:
    """获取模型信息"""
    global _model_session, _model_path
    
    if _model_session is None:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "path": _model_path,
        "providers": _model_session.get_providers(),
        "inputs": [{"name": i.name, "shape": i.shape} for i in _model_session.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape} for o in _model_session.get_outputs()]
    }


# ============ 工具注册表 ============

TOOLS = {
    "load_pointpillars": {
        "func": load_pointpillars,
        "description": "加载 PointPillars 3D 目标检测模型 (ONNX 格式)",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "ONNX 模型文件路径"},
                "device": {"type": "string", "enum": ["cpu", "cuda"], "default": "cpu"}
            },
            "required": ["model_path"]
        }
    },
    "run_pointpillars": {
        "func": run_pointpillars,
        "description": "对当前点云执行 PointPillars 3D 目标检测",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence_threshold": {"type": "number", "default": 0.5, "minimum": 0, "maximum": 1}
            }
        }
    },
    "get_pointpillars_info": {
        "func": get_pointpillars_info,
        "description": "获取 PointPillars 模型信息",
        "parameters": {"type": "object", "properties": {}}
    }
}


# ============ IPC 通信入口 ============

def handle_request(request: dict) -> dict:
    """处理 IPC 请求"""
    tool_name = request.get("tool")
    args = request.get("args", {})
    
    if tool_name not in TOOLS:
        return {
            "result": None,
            "error": f"未知工具：{tool_name}, 可用工具：{list(TOOLS.keys())}"
        }
    
    try:
        func = TOOLS[tool_name]["func"]
        result = func(**args)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": f"工具执行失败：{str(e)}"}


def main():
    """IPC 主循环"""
    np.set_printoptions(precision=6, suppress=True)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            response = handle_request(request)
        except Exception as e:
            response = {"result": None, "error": f"请求处理异常：{str(e)}"}
        
        print(json.dumps(response, ensure_ascii=False), flush=True)


# ============ 本地测试入口 ============

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("=== PointPillars 本地测试 ===\n")
        
        # 1. 加载模型
        print("1. 加载 PointPillars 模型...")
        model_path = os.path.join(os.path.dirname(__file__), "models", "pointpillars.onnx")
        resp = load_pointpillars(model_path, device="cpu")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()
        
        # 2. 创建测试点云
        print("2. 创建测试点云...")
        test_points = np.random.randn(4096, 4).astype(np.float32) * 10
        print(f"   点云形状：{test_points.shape}")
        print()
        
        # 3. 运行推理
        print("3. 运行推理...")
        resp = run_pointpillars(points=test_points, confidence_threshold=0.3)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()
        
        # 4. 获取模型信息
        print("4. 模型信息...")
        resp = get_pointpillars_info()
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()
        
        print("✓ 测试完成")
    else:
        main()
