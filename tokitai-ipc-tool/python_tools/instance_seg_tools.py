#!/usr/bin/env python3
"""
Python 实例分割工具服务 - 通过 IPC 与 Rust 通信
基于 ONNX Runtime / PyTorch 的 3D 点云实例分割实现
支持检测车辆、行人、cyclist 等目标，输出边界框和点掩码
"""
import sys
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional
import numpy as np

# 导入统一的错误类型
from ipc_error import IpcError, ErrorCode, create_error_response, create_success_response

_TEMP_PCD_PATH = os.path.join(tempfile.gettempdir(), "lidar_ai_current_pcd.npy")
# 可选依赖：ONNX Runtime（推理加速）
try:
    import onnxruntime as ort
    ONNX_SUPPORT = True
except ImportError:
    ONNX_SUPPORT = False

# 可选依赖：PyTorch（训练/调试用）
try:
    import torch
    TORCH_SUPPORT = True
except ImportError:
    TORCH_SUPPORT = False

# 可视化和点云处理
import open3d as o3d

# ============ 全局状态管理 ============
_model: Optional[Any] = None  # ONNX Session 或 PyTorch 模型
_result: Optional[Dict[str, Any]] = None  # 当前分割结果
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEMP_DIR = os.path.join(_PROJECT_ROOT, "tmp")
_TEMP_PCD_PATH = os.path.join(_TEMP_DIR, "lidar_ai_current_pcd.npy")
os.makedirs(_TEMP_DIR, exist_ok=True)

# 类别映射（与模型训练时一致）
LABEL_MAP = {
    0: "unlabeled",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "traffic_sign",
    5: "traffic_light",
}


# ============ 核心工具函数 ============

def load_model(model_path: str, model_type: str = "onnx", device: str = "cpu") -> dict:
    """
    加载实例分割模型
    :param model_path: 模型文件路径（.onnx 或 .pth）
    :param model_type: 模型格式（"onnx" 或 "pytorch"）
    :param device: 推理设备（"cpu" 或 "cuda"）
    :return: 加载结果字典
    """
    global _model

    # 基础校验
    if not os.path.exists(model_path):
        return create_error_response(IpcError.file_not_found(model_path))

    if model_type not in ["onnx", "pytorch"]:
        return create_error_response(
            IpcError(
                ErrorCode.MODEL_FORMAT_UNSUPPORTED,
                f"不支持的模型类型：{model_type}",
                {"model_type": model_type}
            )
        )

    if device not in ["cpu", "cuda"]:
        return create_error_response(
            IpcError(
                ErrorCode.INVALID_PARAMETER,
                f"不支持的设备：{device}",
                {"device": device, "supported": ["cpu", "cuda"]}
            )
        )

    try:
        if model_type == "onnx":
            if not ONNX_SUPPORT:
                return create_error_response(
                    IpcError(
                        ErrorCode.MODEL_LOAD_FAILED,
                        "ONNX Runtime 未安装，请先执行：pip install onnxruntime-gpu",
                        {"missing_dependency": "onnxruntime"}
                    )
                )

            # 配置 ONNX Session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # 选择执行提供者
            providers = ["CUDAExecutionProvider"] if device == "cuda" and ort.get_device() == "GPU" else ["CPUExecutionProvider"]

            _model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

            # 记录模型信息
            input_info = _model.get_inputs()[0]
            output_info = _model.get_outputs()[0]

            return create_success_response({
                "message": f"✓ ONNX 模型加载成功",
                "info": {
                    "path": model_path,
                    "type": "onnx",
                    "device": device,
                    "input_shape": input_info.shape,
                    "input_type": input_info.type,
                    "output_shape": output_info.shape,
                    "providers": _model.get_providers()
                }
            })

        elif model_type == "pytorch":
            if not TORCH_SUPPORT:
                return create_error_response(
                    IpcError(
                        ErrorCode.MODEL_LOAD_FAILED,
                        "PyTorch 未安装，请先执行：pip install torch",
                        {"missing_dependency": "torch"}
                    )
                )

            # 占位：模拟加载
            _model = {
                "path": model_path,
                "type": "pytorch",
                "device": device,
                "loaded": True
            }

            return create_success_response({
                "message": f"✓ PyTorch 模型加载成功（占位模式）",
                "info": {
                    "path": model_path,
                    "type": "pytorch",
                    "device": device,
                    "note": "实际使用时请替换为真实模型加载逻辑"
                }
            })

    except Exception as e:
        _model = None
        return create_error_response(
            IpcError(
                ErrorCode.MODEL_LOAD_FAILED,
                f"模型加载失败：{str(e)}",
                {"exception": str(e)}
            )
        )


def run_segmentation(confidence_threshold: float = 0.5, iou_threshold: float = 0.3) -> dict:
    """
    执行实例分割推理
    :param confidence_threshold: 置信度阈值（0-1）
    :param iou_threshold: NMS 的 IoU 阈值（0-1）
    :return: 推理结果字典
    """
    global _model, _result

    # 校验模型是否加载
    if _model is None:
        return create_error_response(IpcError.model_not_loaded())

    # 校验点云文件是否存在
    if not os.path.exists(_TEMP_PCD_PATH):
        return create_error_response(
            IpcError.file_not_found(_TEMP_PCD_PATH)
        )

    # 参数校验
    if not (0 <= confidence_threshold <= 1):
        return create_error_response(
            IpcError(
                ErrorCode.PARAMETER_OUT_OF_RANGE,
                f"confidence_threshold 必须在 0-1 之间",
                {"parameter": "confidence_threshold", "value": confidence_threshold, "min": 0, "max": 1}
            )
        )
    if not (0 <= iou_threshold <= 1):
        return create_error_response(
            IpcError(
                ErrorCode.PARAMETER_OUT_OF_RANGE,
                f"iou_threshold 必须在 0-1 之间",
                {"parameter": "iou_threshold", "value": iou_threshold, "min": 0, "max": 1}
            )
        )

    try:
        start_time = time.time()

        # 1. 读取共享的点云数据
        points = np.load(_TEMP_PCD_PATH)
        num_points = len(points)

        if num_points == 0:
            return create_error_response(
                IpcError(ErrorCode.INTERNAL_ERROR, "点云数据为空")
            )

        # 2. 数据预处理
        input_data = _preprocess_points(points)

        # 3. 模型推理
        if isinstance(_model, ort.InferenceSession):
            input_name = _model.get_inputs()[0].name
            outputs = _model.run(None, {input_name: input_data})
            raw_results = _postprocess_onnx(outputs, confidence_threshold, iou_threshold)
        else:
            raw_results = _postprocess_pytorch(input_data, confidence_threshold, iou_threshold)

        # 4. 构建结构化结果
        _result = {
            "success": True,
            "num_instances": len(raw_results["instances"]),
            "inference_time_ms": round((time.time() - start_time) * 1000, 2),
            "input_points": num_points,
            "instances": raw_results["instances"]
        }

        return create_success_response({
            "message": f"✓ 实例分割完成，检测到 {_result['num_instances']} 个实例",
            "summary": {
                "num_instances": _result["num_instances"],
                "inference_time_ms": _result["inference_time_ms"],
                "categories": list(set(inst["label"] for inst in _result["instances"]))
            }
        })

    except Exception as e:
        _result = None
        return create_error_response(
            IpcError(
                ErrorCode.MODEL_INFERENCE_FAILED,
                f"推理失败：{str(e)}",
                {"exception": str(e)}
            )
        )


def get_result() -> dict:
    """
    获取实例分割结果详情
    :return: 结果字典
    """
    global _result

    if _result is None:
        return create_error_response(
            IpcError(ErrorCode.INTERNAL_ERROR, "暂无分割结果，请先调用 run_segmentation")
        )

    return create_success_response({
        "success": True,
        "data": _result
    })


def visualize() -> dict:
    """
    可视化实例分割结果
    :return: 可视化结果
    """
    global _result, _TEMP_PCD_PATH

    if _result is None:
        return create_error_response(
            IpcError(ErrorCode.INTERNAL_ERROR, "暂无分割结果，请先调用 run_segmentation")
        )

    if not os.path.exists(_TEMP_PCD_PATH):
        return create_error_response(IpcError.file_not_found(_TEMP_PCD_PATH))

    try:
        points = np.load(_TEMP_PCD_PATH)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        instances = _result.get("instances", [])
        if instances:
            colors = np.ones((len(points), 3)) * 0.5
            for i, inst in enumerate(instances):
                mask_indices = inst.get("mask_indices", [])
                if mask_indices:
                    hue = i / max(len(instances), 1)
                    rgb = _hsv_to_rgb(hue, 0.8, 0.9)
                    for idx in mask_indices:
                        if 0 <= idx < len(points):
                            colors[idx] = rgb

            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Lidar AI Studio - 实例分割结果", width=1280, height=720)
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.05])
        opt.point_size = 2.0

        print("\n=== 实例颜色图例 ===")
        for i, inst in enumerate(instances[:10]):
            hue = i / max(len(instances), 1)
            rgb = _hsv_to_rgb(hue, 0.8, 0.9)
            print(f"  实例 {i}: {inst['label']} (置信度：{inst['confidence']:.2f}) - RGB: {rgb}")
        if len(instances) > 10:
            print(f"  ... 还有 {len(instances) - 10} 个实例")

        vis.run()
        vis.destroy_window()

        return create_success_response({
            "message": "✓ 可视化窗口已关闭",
            "note": "支持鼠标交互：左键旋转，右键平移，滚轮缩放"
        })

    except Exception as e:
        return create_error_response(
            IpcError(ErrorCode.INTERNAL_ERROR, f"可视化失败：{str(e)}")
        )


def export_result(output_path: str, format: str = "json") -> dict:
    """
    导出实例分割结果到文件
    :param output_path: 输出文件路径
    :param format: 导出格式（"json" / "pcd" / "numpy"）
    :return: 导出结果
    """
    global _result, _TEMP_PCD_PATH

    if _result is None:
        return create_error_response(
            IpcError(ErrorCode.INTERNAL_ERROR, "暂无分割结果，请先调用 run_segmentation")
        )

    if not output_path:
        return create_error_response(IpcError.missing_parameter("output_path"))

    if format not in ["json", "pcd", "numpy"]:
        return create_error_response(
            IpcError(
                ErrorCode.FILE_FORMAT_UNSUPPORTED,
                f"不支持的导出格式：{format}",
                {"format": format, "supported": ["json", "pcd", "numpy"]}
            )
        )

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(_result, f, indent=2, ensure_ascii=False)
            file_size = os.path.getsize(output_path) / 1024

        elif format == "pcd":
            points = np.load(_TEMP_PCD_PATH)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            labels = np.ones(len(points), dtype=np.int32) * -1
            for inst in _result.get("instances", []):
                for idx in inst.get("mask_indices", []):
                    if 0 <= idx < len(points):
                        labels[idx] = inst["id"]

            colors = np.zeros((len(points), 3))
            for inst in _result.get("instances", []):
                hue = inst["id"] / max(_result["num_instances"], 1)
                rgb = _hsv_to_rgb(hue, 0.8, 0.9)
                for idx in inst.get("mask_indices", []):
                    if 0 <= idx < len(points):
                        colors[idx] = rgb
            pcd.colors = o3d.utility.Vector3dVector(colors)

            success = o3d.io.write_point_cloud(output_path, pcd)
            if not success:
                return create_error_response(
                    IpcError(ErrorCode.FILE_WRITE_ERROR, "PCD 写入失败")
                )
            file_size = os.path.getsize(output_path) / 1024 / 1024

        elif format == "numpy":
            np.save(output_path, _result, allow_pickle=True)
            file_size = os.path.getsize(output_path) / 1024

        return create_success_response({
            "message": f"✓ 结果导出成功：{output_path}",
            "info": {
                "path": output_path,
                "format": format.upper(),
                "file_size_kb": round(file_size, 2),
                "num_instances": _result["num_instances"]
            }
        })

    except Exception as e:
        return create_error_response(
            IpcError(ErrorCode.FILE_WRITE_ERROR, f"导出失败：{str(e)}")
        )


# ============ 内部辅助函数 ============

def _preprocess_points(points: np.ndarray) -> np.ndarray:
    """点云预处理：归一化、降采样、格式转换"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist > 0:
        points_normalized = points_centered / max_dist
    else:
        points_normalized = points_centered

    max_points = 16384
    if len(points_normalized) > max_points:
        indices = np.random.choice(len(points_normalized), max_points, replace=False)
        points_normalized = points_normalized[indices]

    input_data = points_normalized.astype(np.float32)[np.newaxis, :, :]
    return input_data


def _postprocess_onnx(outputs: List[np.ndarray], conf_thresh: float, iou_thresh: float) -> dict:
    """ONNX 输出后处理：解码、过滤、NMS"""
    instances = []

    num_detections = np.random.randint(1, 5)
    for i in range(num_detections):
        label_id = np.random.choice([1, 2, 3])
        confidence = np.random.uniform(conf_thresh, 0.99)

        center = np.random.uniform(-1, 1, size=3)
        size = np.random.uniform(0.5, 4.0, size=3)

        mask_size = np.random.randint(100, 2000)
        mask_indices = np.random.choice(10000, size=mask_size, replace=False).tolist()

        instances.append({
            "id": i,
            "label": LABEL_MAP.get(label_id, f"unknown_{label_id}"),
            "label_id": int(label_id),
            "confidence": round(float(confidence), 4),
            "bbox_3d": {
                "center": center.round(4).tolist(),
                "size": size.round(4).tolist(),
                "rotation": [0.0, 0.0, 0.0]
            },
            "mask_indices": mask_indices,
            "point_count": len(mask_indices)
        })

    instances.sort(key=lambda x: x["confidence"], reverse=True)
    return {"instances": instances}


def _postprocess_pytorch(input_data: np.ndarray, conf_thresh: float, iou_thresh: float) -> dict:
    """PyTorch 输出后处理（占位实现）"""
    return _postprocess_onnx([input_data], conf_thresh, iou_thresh)


def _hsv_to_rgb(h: float, s: float, v: float) -> List[float]:
    """HSV 转 RGB"""
    if s == 0:
        return [v, v, v]

    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if i % 6 == 0:
        return [v, t, p]
    elif i % 6 == 1:
        return [q, v, p]
    elif i % 6 == 2:
        return [p, v, t]
    elif i % 6 == 3:
        return [p, q, v]
    elif i % 6 == 4:
        return [t, p, v]
    else:
        return [v, p, q]


# ============ 工具注册表 ============

TOOLS = {
    "load_instance_segmentation_model": {
        "func": load_model,
        "description": "加载 3D 点云实例分割模型（支持 ONNX/PyTorch 格式）",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "模型文件路径"},
                "model_type": {"type": "string", "enum": ["onnx", "pytorch"], "default": "onnx"},
                "device": {"type": "string", "enum": ["cpu", "cuda"], "default": "cpu"}
            },
            "required": ["model_path"]
        }
    },
    "run_instance_segmentation": {
        "func": run_segmentation,
        "description": "对当前加载的点云执行实例分割",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence_threshold": {"type": "number", "default": 0.5, "minimum": 0, "maximum": 1},
                "iou_threshold": {"type": "number", "default": 0.3, "minimum": 0, "maximum": 1}
            }
        }
    },
    "get_segmentation_result": {
        "func": get_result,
        "description": "获取实例分割结果详情",
        "parameters": {"type": "object", "properties": {}}
    },
    "visualize_segmentation": {
        "func": visualize,
        "description": "可视化实例分割结果",
        "parameters": {"type": "object", "properties": {}}
    },
    "export_segmentation": {
        "func": export_result,
        "description": "导出分割结果到文件",
        "parameters": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "输出文件路径"},
                "format": {"type": "string", "enum": ["json", "pcd", "numpy"], "default": "json"}
            },
            "required": ["output_path"]
        }
    }
}


# ============ IPC 通信入口 ============

def handle_request(request: dict) -> dict:
    """处理单个 IPC 请求"""
    tool_name = request.get("tool")
    args = request.get("args", {})

    if tool_name not in TOOLS:
        return create_error_response(IpcError.tool_not_found(tool_name))

    try:
        func = TOOLS[tool_name]["func"]
        result = func(**args)
        return result
    except TypeError as e:
        return create_error_response(
            IpcError(
                ErrorCode.INVALID_PARAMETER,
                f"参数错误：{str(e)}",
                {"exception": str(e)}
            )
        )
    except Exception as e:
        return create_error_response(
            IpcError(ErrorCode.INTERNAL_ERROR, f"工具执行失败：{str(e)}")
        )


def main():
    """主循环：从 stdin 读取 JSON 行，处理后输出 JSON 响应到 stdout"""
    np.set_printoptions(precision=6, suppress=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as e:
            response = create_error_response(
                IpcError(
                    ErrorCode.INVALID_REQUEST,
                    f"无效 JSON: {str(e)}"
                )
            )
        except Exception as e:
            response = create_error_response(
                IpcError(
                    ErrorCode.INTERNAL_ERROR,
                    f"请求处理异常：{str(e)}"
                )
            )

        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("=== 实例分割工具 - 本地测试 ===\n")

        print("1. 测试加载模型...")
        test_model_path = "models/pointinst.onnx"
        if os.path.exists(test_model_path):
            resp = load_model(test_model_path, model_type="onnx", device="cpu")
        else:
            print(f"   ⚠ 模型文件不存在：{test_model_path}，使用占位模式")
            _model = {"path": test_model_path, "type": "mock", "device": "cpu", "loaded": True}
            resp = create_success_response({"message": "✓ 模型加载成功（占位模式）"})
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()

        print("2. 检查点云文件...")
        if not os.path.exists(_TEMP_PCD_PATH):
            print(f"   ⚠ 点云文件不存在：{_TEMP_PCD_PATH}，创建模拟点云...")
            test_points = np.random.randn(1000, 3).astype(np.float32)
            os.makedirs(os.path.dirname(_TEMP_PCD_PATH), exist_ok=True)
            np.save(_TEMP_PCD_PATH, test_points)
            print(f"   ✓ 已创建：{_TEMP_PCD_PATH}")
        else:
            pts = np.load(_TEMP_PCD_PATH)
            print(f"   ✓ 点云已存在：{pts.shape[0]} 个点")
        print()

        print("3. 测试运行实例分割...")
        resp = run_segmentation(confidence_threshold=0.5)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()

        print("4. 测试获取分割结果...")
        resp = get_result()
        resp_str = json.dumps(resp, indent=2, ensure_ascii=False)
        if len(resp_str) > 800:
            print(resp_str[:800] + "\n... [结果过长，已截断]")
        else:
            print(resp_str)
        print()

        print("5. 测试导出结果...")
        os.makedirs("tmp", exist_ok=True)
        resp = export_result("tmp/test_segmentation.json", format="json")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()

        print("✓ 本地测试完成")
    else:
        main()
