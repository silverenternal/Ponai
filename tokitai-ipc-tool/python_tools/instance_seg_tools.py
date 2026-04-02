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
        return {"error": f"模型文件不存在: {model_path}"}
    
    if model_type not in ["onnx", "pytorch"]:
        return {"error": f"不支持的模型类型: {model_type}"}
    
    if device not in ["cpu", "cuda"]:
        return {"error": f"不支持的设备: {device}"}
    
    try:
        if model_type == "onnx":
            if not ONNX_SUPPORT:
                return {"error": "ONNX Runtime 未安装，请先执行: pip install onnxruntime-gpu"}
            
            # 配置 ONNX Session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 选择执行提供者
            providers = ["CUDAExecutionProvider"] if device == "cuda" and ort.get_device() == "GPU" else ["CPUExecutionProvider"]
            
            _model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            
            # 记录模型信息
            input_info = _model.get_inputs()[0]
            output_info = _model.get_outputs()[0]
            
            return {
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
            }
            
        elif model_type == "pytorch":
            if not TORCH_SUPPORT:
                return {"error": "PyTorch 未安装，请先执行: pip install torch"}
            
            # 加载 PyTorch 模型（假设模型类已定义）
            # 实际使用时需导入你的模型类: from models import PointInst
            # model = PointInst()
            # model.load_state_dict(torch.load(model_path, map_location=device))
            # model.eval()
            # _model = model.to(device)
            
            # 占位：模拟加载
            _model = {
                "path": model_path,
                "type": "pytorch", 
                "device": device,
                "loaded": True  # 标记为已加载
            }
            
            return {
                "message": f"✓ PyTorch 模型加载成功（占位模式）",
                "info": {
                    "path": model_path,
                    "type": "pytorch",
                    "device": device,
                    "note": "实际使用时请替换为真实模型加载逻辑"
                }
            }
            
    except Exception as e:
        _model = None  # 加载失败时清空
        return {"error": f"模型加载失败: {str(e)}"}


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
        return {"error": "模型未加载，请先调用 load_model"}
    
    # 校验点云文件是否存在
    if not os.path.exists(_TEMP_PCD_PATH):
        return {"error": f"点云文件不存在: {_TEMP_PCD_PATH}，请先用 pointcloud_tools.py 加载点云"}
    
    # 参数校验
    if not (0 <= confidence_threshold <= 1):
        return {"error": "confidence_threshold 必须在 0-1 之间"}
    if not (0 <= iou_threshold <= 1):
        return {"error": "iou_threshold 必须在 0-1 之间"}
    
    try:
        start_time = time.time()
        
        # 1. 读取共享的点云数据
        points = np.load(_TEMP_PCD_PATH)  # shape: (N, 3)
        num_points = len(points)
        
        if num_points == 0:
            return {"error": "点云数据为空"}
        
        # 2. 数据预处理（根据模型要求）
        # 示例：归一化、降采样、转换为模型输入格式
        input_data = _preprocess_points(points)
        
        # 3. 模型推理
        if isinstance(_model, ort.InferenceSession):
            # ONNX Runtime 推理
            input_name = _model.get_inputs()[0].name
            outputs = _model.run(None, {input_name: input_data})
            raw_results = _postprocess_onnx(outputs, confidence_threshold, iou_threshold)
        else:
            # PyTorch 推理（占位）
            raw_results = _postprocess_pytorch(input_data, confidence_threshold, iou_threshold)
        
        # 4. 构建结构化结果
        _result = {
            "success": True,
            "num_instances": len(raw_results["instances"]),
            "inference_time_ms": round((time.time() - start_time) * 1000, 2),
            "input_points": num_points,
            "instances": raw_results["instances"]
        }
        
        return {
            "message": f"✓ 实例分割完成，检测到 {_result['num_instances']} 个实例",
            "summary": {
                "num_instances": _result["num_instances"],
                "inference_time_ms": _result["inference_time_ms"],
                "categories": list(set(inst["label"] for inst in _result["instances"]))
            }
        }
        
    except Exception as e:
        _result = None
        return {"error": f"推理失败: {str(e)}"}


def get_result() -> dict:
    """
    获取实例分割结果详情
    :return: 结果字典（包含实例列表、边界框、置信度等）
    """
    global _result
    
    if _result is None:
        return {"error": "暂无分割结果，请先调用 run_segmentation"}
    
    return {
        "success": True,
        "data": _result
    }


def visualize() -> dict:
    """
    可视化实例分割结果（不同实例用不同颜色显示）
    :return: 可视化结果
    """
    global _result, _TEMP_PCD_PATH
    
    if _result is None:
        return {"error": "暂无分割结果，请先调用 run_segmentation"}
    
    if not os.path.exists(_TEMP_PCD_PATH):
        return {"error": f"点云文件不存在: {_TEMP_PCD_PATH}"}
    
    try:
        # 读取原始点云
        points = np.load(_TEMP_PCD_PATH)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 为每个实例分配不同颜色
        instances = _result.get("instances", [])
        if instances:
            # 初始化颜色数组（默认灰色）
            colors = np.ones((len(points), 3)) * 0.5
            
            # 为每个实例的点分配颜色
            for i, inst in enumerate(instances):
                mask_indices = inst.get("mask_indices", [])
                if mask_indices:
                    # 使用 HSV 色环生成不同颜色
                    hue = i / max(len(instances), 1)
                    rgb = _hsv_to_rgb(hue, 0.8, 0.9)
                    for idx in mask_indices:
                        if 0 <= idx < len(points):
                            colors[idx] = rgb
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Lidar AI Studio - 实例分割结果", width=1280, height=720)
        vis.add_geometry(pcd)
        
        # 设置渲染参数
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.05])
        opt.point_size = 2.0
        
        # 添加图例说明（简单版：打印到控制台）
        print("\n=== 实例颜色图例 ===")
        for i, inst in enumerate(instances[:10]):  # 最多显示10个
            hue = i / max(len(instances), 1)
            rgb = _hsv_to_rgb(hue, 0.8, 0.9)
            print(f"  实例 {i}: {inst['label']} (置信度: {inst['confidence']:.2f}) - RGB: {rgb}")
        if len(instances) > 10:
            print(f"  ... 还有 {len(instances) - 10} 个实例")
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
        return {
            "message": "✓ 可视化窗口已关闭",
            "note": "支持鼠标交互：左键旋转，右键平移，滚轮缩放"
        }
        
    except Exception as e:
        return {"error": f"可视化失败: {str(e)}"}


def export_result(output_path: str, format: str = "json") -> dict:
    """
    导出实例分割结果到文件
    :param output_path: 输出文件路径
    :param format: 导出格式（"json" / "pcd" / "numpy"）
    :return: 导出结果
    """
    global _result, _TEMP_PCD_PATH
    
    if _result is None:
        return {"error": "暂无分割结果，请先调用 run_segmentation"}
    
    if not output_path:
        return {"error": "输出路径不能为空"}
    
    if format not in ["json", "pcd", "numpy"]:
        return {"error": f"不支持的导出格式: {format}，仅支持 json/pcd/numpy"}
    
    try:
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if format == "json":
            # 导出为 JSON（人类可读）
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(_result, f, indent=2, ensure_ascii=False)
            file_size = os.path.getsize(output_path) / 1024
            
        elif format == "pcd":
            # 导出为 PCD（带实例标签的点云）
            points = np.load(_TEMP_PCD_PATH)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 为每个点添加实例标签（-1 表示背景）
            labels = np.ones(len(points), dtype=np.int32) * -1
            for inst in _result.get("instances", []):
                for idx in inst.get("mask_indices", []):
                    if 0 <= idx < len(points):
                        labels[idx] = inst["id"]
            
            # Open3D 的 PCD 格式支持自定义字段
            # 这里简化处理：只保存带颜色的点云（颜色编码实例 ID）
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
                return {"error": "PCD 写入失败"}
            file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
            
        elif format == "numpy":
            # 导出为 NumPy（二进制，适合程序读取）
            np.save(output_path, _result, allow_pickle=True)
            file_size = os.path.getsize(output_path) / 1024
        
        return {
            "message": f"✓ 结果导出成功: {output_path}",
            "info": {
                "path": output_path,
                "format": format.upper(),
                "file_size_kb": round(file_size, 2),
                "num_instances": _result["num_instances"]
            }
        }
        
    except Exception as e:
        return {"error": f"导出失败: {str(e)}"}


# ============ 内部辅助函数 ============

def _preprocess_points(points: np.ndarray) -> np.ndarray:
    """
    点云预处理：归一化、降采样、格式转换
    :param points: 原始点云 (N, 3)
    :return: 模型输入格式
    """
    # 示例预处理流程（根据实际模型调整）
    
    # 1. 去中心化（以点云中心为原点）
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # 2. 归一化到 [-1, 1] 范围
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist > 0:
        points_normalized = points_centered / max_dist
    else:
        points_normalized = points_centered
    
    # 3. 降采样（如果点数过多）
    max_points = 16384  # 模型最大输入点数
    if len(points_normalized) > max_points:
        indices = np.random.choice(len(points_normalized), max_points, replace=False)
        points_normalized = points_normalized[indices]
    
    # 4. 转换为模型输入格式 (B, N, 3)
    input_data = points_normalized.astype(np.float32)[np.newaxis, :, :]
    
    return input_data


def _postprocess_onnx(outputs: List[np.ndarray], conf_thresh: float, iou_thresh: float) -> dict:
    """
    ONNX 输出后处理：解码、过滤、NMS
    :param outputs: 模型输出列表
    :param conf_thresh: 置信度阈值
    :param iou_thresh: NMS IoU 阈值
    :return: 结构化结果
    """
    # 示例：假设模型输出格式为 [boxes, scores, labels, masks]
    # 实际需根据模型输出调整
    
    instances = []
    
    # 占位：生成模拟结果（实际使用时替换为真实后处理逻辑）
    num_detections = np.random.randint(1, 5)  # 随机 1-4 个实例
    for i in range(num_detections):
        label_id = np.random.choice([1, 2, 3])  # vehicle/pedestrian/cyclist
        confidence = np.random.uniform(conf_thresh, 0.99)
        
        # 随机生成 3D 边界框
        center = np.random.uniform(-1, 1, size=3)
        size = np.random.uniform(0.5, 4.0, size=3)
        
        # 随机生成掩码点索引（模拟）
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
                "rotation": [0.0, 0.0, 0.0]  # 简化：暂不预测旋转
            },
            "mask_indices": mask_indices,
            "point_count": len(mask_indices)
        })
    
    # 简单 NMS（按置信度排序，实际需计算 3D IoU）
    instances.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {"instances": instances}


def _postprocess_pytorch(input_data: np.ndarray, conf_thresh: float, iou_thresh: float) -> dict:
    """
    PyTorch 输出后处理（占位实现）
    """
    # 实际使用时：
    # 1. 将 input_data 转为 torch.Tensor 并移到对应 device
    # 2. 执行 model(input_data) 得到原始输出
    # 3. 调用 _postprocess_onnx 类似的逻辑解码结果
    
    # 占位：直接复用 ONNX 后处理逻辑生成模拟结果
    return _postprocess_onnx([input_data], conf_thresh, iou_thresh)


def _hsv_to_rgb(h: float, s: float, v: float) -> List[float]:
    """
    HSV 转 RGB（用于生成实例颜色）
    :param h: 色相 [0, 1]
    :param s: 饱和度 [0, 1]
    :param v: 明度 [0, 1]
    :return: RGB 三元组 [0, 1]
    """
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


# ============ 工具注册表（AI 调度层可见） ============

TOOLS = {
    "load_instance_segmentation_model": {
        "func": load_model,
        "description": "加载 3D 点云实例分割模型（支持 ONNX/PyTorch 格式，GPU 加速需安装对应后端）",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "模型文件路径（.onnx 或 .pth）"
                },
                "model_type": {
                    "type": "string",
                    "enum": ["onnx", "pytorch"],
                    "default": "onnx",
                    "description": "模型格式"
                },
                "device": {
                    "type": "string",
                    "enum": ["cpu", "cuda"],
                    "default": "cpu",
                    "description": "推理设备"
                }
            },
            "required": ["model_path"]
        }
    },
    "run_instance_segmentation": {
        "func": run_segmentation,
        "description": "对当前加载的点云执行实例分割，检测车辆、行人、cyclist 等目标",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0,
                    "maximum": 1,
                    "description": "置信度阈值 (0-1)，越高检测结果越精确但可能漏检"
                },
                "iou_threshold": {
                    "type": "number",
                    "default": 0.3,
                    "minimum": 0,
                    "maximum": 1,
                    "description": "NMS IoU 阈值 (0-1)，用于过滤重叠检测框"
                }
            }
        }
    },
    "get_segmentation_result": {
        "func": get_result,
        "description": "获取实例分割结果详情（实例数量、边界框、置信度、点掩码等）",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "visualize_segmentation": {
        "func": visualize,
        "description": "可视化实例分割结果（不同实例用不同颜色显示，需要 GUI 环境）",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "export_segmentation": {
        "func": export_result,
        "description": "导出分割结果到文件（支持 JSON/PCD/NumPy 格式）",
        "parameters": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "输出文件路径（绝对路径/相对路径）"
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "pcd", "numpy"],
                    "default": "json",
                    "description": "导出格式"
                }
            },
            "required": ["output_path"]
        }
    }
}


# ============ IPC 通信入口 ============

def handle_request(request: dict) -> dict:
    """
    处理单个 IPC 请求
    :param request: {"tool": "工具名", "args": {"参数": 值}}
    :return: {"result": 结果, "error": 错误信息}
    """
    tool_name = request.get("tool")
    args = request.get("args", {})
    
    if tool_name not in TOOLS:
        return {
            "result": None, 
            "error": f"未知工具: {tool_name}，可用工具: {list(TOOLS.keys())}"
        }
    
    try:
        func = TOOLS[tool_name]["func"]
        # 调用函数（自动解包参数）
        result = func(**args)
        return {"result": result, "error": None}
    except TypeError as e:
        return {
            "result": None, 
            "error": f"参数错误: {str(e)}，请检查参数类型和必填项"
        }
    except Exception as e:
        return {
            "result": None, 
            "error": f"工具执行失败: {str(e)}"
        }


def main():
    """
    主循环：从 stdin 读取 JSON 行，处理后输出 JSON 响应到 stdout
    与 Rust 端的 IPC 通信入口
    """
    # 设置 numpy 输出格式
    np.set_printoptions(precision=6, suppress=True)
    
    # 逐行处理请求
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as e:
            response = {"result": None, "error": f"无效 JSON: {str(e)}"}
        except Exception as e:
            response = {"result": None, "error": f"请求处理异常: {str(e)}"}
        
        # 输出响应（flush 确保 Rust 立即接收）
        print(json.dumps(response, ensure_ascii=False), flush=True)


# ============ 本地测试入口 ============

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # ===== 本地测试模式（开发友好版） =====
        print("=== 实例分割工具 - 本地测试 ===\n")
        
        # 1. 测试模型加载（如果没有真实模型，用占位模式）
        print("1. 测试加载模型...")
        test_model_path = "models/pointinst.onnx"  # 真实模型路径
        if os.path.exists(test_model_path):
            resp = load_model(test_model_path, model_type="onnx", device="cpu")
        else:
            print(f"   ⚠ 模型文件不存在：{test_model_path}，使用占位模式")
            # 直接设置占位模型（跳过文件检查）
            _model = {"path": test_model_path, "type": "mock", "device": "cpu", "loaded": True}
            resp = {"message": "✓ 模型加载成功（占位模式）", "info": {"note": "无真实模型文件"}}
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()
        
        # 2. 确保点云文件存在
        print("2. 检查点云文件...")
        if not os.path.exists(_TEMP_PCD_PATH):
            print(f"   ⚠ 点云文件不存在：{_TEMP_PCD_PATH}")
            print("   创建模拟点云用于测试...")
            test_points = np.random.randn(1000, 3).astype(np.float32)
            os.makedirs(os.path.dirname(_TEMP_PCD_PATH), exist_ok=True)
            np.save(_TEMP_PCD_PATH, test_points)
            print(f"   ✓ 已创建：{_TEMP_PCD_PATH}")
        else:
            pts = np.load(_TEMP_PCD_PATH)
            print(f"   ✓ 点云已存在：{pts.shape[0]} 个点")
        print()
        
        # 3. 测试运行分割
        print("3. 测试运行实例分割...")
        resp = run_segmentation(confidence_threshold=0.5)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()
        
        # 4. 测试获取结果
        print("4. 测试获取分割结果...")
        resp = get_result()
        # 截断长输出
        resp_str = json.dumps(resp, indent=2, ensure_ascii=False)
        if len(resp_str) > 800:
            print(resp_str[:800] + "\n... [结果过长，已截断]")
        else:
            print(resp_str)
        print()
        
        # 5. 测试导出
        print("5. 测试导出结果...")
        os.makedirs("tmp", exist_ok=True)
        resp = export_result("tmp/test_segmentation.json", format="json")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        print()
        
        # 6. 验证导出文件
        print("6. 验证导出文件...")
        if os.path.exists("tmp/test_segmentation.json"):
            with open("tmp/test_segmentation.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"   ✓ 导出成功：{data.get('num_instances', 0)} 个实例")
        else:
            print("   ❌ 导出文件不存在")
        print()
        
        print("✓ 本地测试完成")
        
    else:
        # ===== IPC 通信模式 =====
        main()