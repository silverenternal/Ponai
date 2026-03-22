"""
Python 点云工具服务 - 通过 IPC 与 Rust 通信
基于 Open3D 的完整点云处理工具实现
支持 PCD/PLY/LAS 格式，包含加载、降采样、法线估计、聚类等核心功能
"""
import sys
import json
import os
from typing import Any, Dict, Optional
import numpy as np
import open3d as o3d

# 全局点云数据管理（线程不安全，生产环境可改用类封装）
_point_cloud_data: Optional[o3d.geometry.PointCloud] = None
_point_cloud_info: Dict[str, Any] = {}
# LAS 格式支持（需额外安装 laspy）
try:
    import laspy
    LAS_SUPPORT = True
except ImportError:
    LAS_SUPPORT = False


def load_point_cloud(file_path: str) -> dict:
    """
    加载点云文件（支持 PCD、PLY、LAS 格式）
    :param file_path: 点云文件路径
    :return: 加载结果字典（JSON 可序列化）
    """
    global _point_cloud_data, _point_cloud_info
    
    # 基础校验
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}
    
    file_ext = file_path.split(".")[-1].lower()
    _point_cloud_info = {
        "file_path": file_path,
        "format": file_ext.upper(),
        "loaded": False,
        "las_support": LAS_SUPPORT
    }

    try:
        # 加载不同格式的点云
        if file_ext in ["pcd", "ply"]:
            _point_cloud_data = o3d.io.read_point_cloud(file_path)
        elif file_ext == "las" and LAS_SUPPORT:
            # LAS 格式转换为 Open3D 点云
            las = laspy.read(file_path)
            points = np.vstack((las.x, las.y, las.z)).T
            _point_cloud_data = o3d.geometry.PointCloud()
            _point_cloud_data.points = o3d.utility.Vector3dVector(points)
            # 保留颜色信息（如果有）
            if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
                colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
                _point_cloud_data.colors = o3d.utility.Vector3dVector(colors)
        elif file_ext == "las" and not LAS_SUPPORT:
            return {"error": "LAS 格式需要安装 laspy: pip install laspy[laszip]"}
        else:
            return {"error": f"不支持的文件格式: {file_ext}，仅支持 PCD/PLY/LAS"}

        # 更新点云基础信息
        points = np.asarray(_point_cloud_data.points)
        _point_cloud_info.update({
            "loaded": True,
            "num_points": len(points),
            "has_normals": _point_cloud_data.has_normals(),
            "has_colors": _point_cloud_data.has_colors(),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
            "bbox_size": (points.max(axis=0) - points.min(axis=0)).tolist()
        })

        return {
            "message": f"成功加载点云文件：{file_path}",
            "info": _point_cloud_info
        }

    except Exception as e:
        return {"error": f"加载失败: {str(e)}"}


def get_point_cloud_info() -> dict:
    """
    获取点云完整信息（点数、边界框、密度、法线/颜色状态等）
    :return: 点云信息字典
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    try:
        points = np.asarray(_point_cloud_data.points)
        # 计算点云密度（单位体积内的点数）
        bbox_size = np.array(_point_cloud_info["bbox_size"])
        volume = bbox_size[0] * bbox_size[1] * bbox_size[2] if all(bbox_size) > 0 else 1.0
        density = len(points) / volume

        # 完整信息
        info = {
            "basic": {
                "num_points": len(points),
                "format": _point_cloud_info["format"],
                "file_path": _point_cloud_info["file_path"]
            },
            "geometry": {
                "bounding_box": {
                    "min": _point_cloud_info["bbox_min"],
                    "max": _point_cloud_info["bbox_max"],
                    "size": _point_cloud_info["bbox_size"],
                    "center": np.mean(points, axis=0).tolist()
                },
                "density": round(density, 6),
                "dimension": 3  # 3D 点云固定为 3
            },
            "attributes": {
                "has_normals": _point_cloud_data.has_normals(),
                "has_colors": _point_cloud_data.has_colors(),
                "has_curvature": False  # Open3D 点云默认无曲率信息
            },
            "statistics": {
                "mean": np.mean(points, axis=0).tolist(),
                "std": np.std(points, axis=0).tolist(),
                "median": np.median(points, axis=0).tolist()
            }
        }

        return info

    except Exception as e:
        return {"error": f"获取信息失败: {str(e)}"}


def downsample(voxel_size: float) -> dict:
    """
    体素网格降采样（保留点云整体结构，减少点数）
    :param voxel_size: 体素网格大小（越大降采样越明显）
    :return: 降采样结果
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    if voxel_size <= 0:
        return {"error": "体素大小必须大于 0"}

    try:
        # 执行降采样
        original_points = len(np.asarray(_point_cloud_data.points))
        _point_cloud_data = _point_cloud_data.voxel_down_sample(voxel_size=voxel_size)
        new_points = len(np.asarray(_point_cloud_data.points))

        # 更新全局信息
        _point_cloud_info["num_points"] = new_points
        _point_cloud_info["downsampled"] = True
        _point_cloud_info["voxel_size"] = voxel_size

        return {
            "message": f"降采样完成（体素大小: {voxel_size}）",
            "statistics": {
                "original_points": original_points,
                "new_points": new_points,
                "reduction_rate": round(1 - new_points/original_points, 4) * 100,
                "voxel_size": voxel_size
            }
        }

    except Exception as e:
        return {"error": f"降采样失败: {str(e)}"}


def estimate_normals(k_neighbors: int) -> dict:
    """
    法线估计（基于 K 近邻）
    :param k_neighbors: 邻域点数（建议 10-30）
    :return: 法线估计结果
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    if k_neighbors < 3:
        return {"error": "邻域点数必须大于等于 3"}

    try:
        # 执行法线估计
        _point_cloud_data.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        # 法线方向一致性调整
        _point_cloud_data.orient_normals_towards_camera_location()

        # 更新全局信息
        _point_cloud_info["has_normals"] = True
        _point_cloud_info["normals_k_neighbors"] = k_neighbors

        normals = np.asarray(_point_cloud_data.normals)
        return {
            "message": f"法线估计完成（邻域点数: {k_neighbors}）",
            "statistics": {
                "k_neighbors": k_neighbors,
                "normals_count": len(normals),
                "normals_range": {
                    "min": normals.min(axis=0).tolist(),
                    "max": normals.max(axis=0).tolist()
                }
            }
        }

    except Exception as e:
        return {"error": f"法线估计失败: {str(e)}"}


def remove_outliers(nb_neighbors: int, std_ratio: float) -> dict:
    """
    统计离群点移除（基于邻域距离的标准差过滤）
    :param nb_neighbors: 邻域点数（建议 20-50）
    :param std_ratio: 标准差比率（建议 1.0-2.0）
    :return: 离群点移除结果
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    if nb_neighbors < 1:
        return {"error": "邻域点数必须大于 0"}
    if std_ratio <= 0:
        return {"error": "标准差比率必须大于 0"}

    try:
        # 执行离群点移除
        original_points = len(np.asarray(_point_cloud_data.points))
        filtered_cloud, ind = _point_cloud_data.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        _point_cloud_data = filtered_cloud
        new_points = len(np.asarray(_point_cloud_data.points))

        # 更新全局信息
        _point_cloud_info["num_points"] = new_points
        _point_cloud_info["outliers_removed"] = True

        return {
            "message": f"离群点移除完成",
            "statistics": {
                "original_points": original_points,
                "remaining_points": new_points,
                "removed_points": original_points - new_points,
                "removal_rate": round((original_points - new_points)/original_points * 100, 2),
                "parameters": {
                    "nb_neighbors": nb_neighbors,
                    "std_ratio": std_ratio
                }
            }
        }

    except Exception as e:
        return {"error": f"离群点移除失败: {str(e)}"}


def segment_plane(distance_threshold: float, max_iterations: int) -> dict:
    """
    平面分割（RANSAC 算法，常用于地面分割）
    :param distance_threshold: 距离阈值（建议 0.01-0.1）
    :param max_iterations: 最大迭代次数（建议 1000-10000）
    :return: 平面分割结果
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    if distance_threshold <= 0:
        return {"error": "距离阈值必须大于 0"}
    if max_iterations < 1:
        return {"error": "最大迭代次数必须大于 0"}

    try:
        # 执行平面分割
        plane_model, inliers = _point_cloud_data.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=max_iterations
        )
        [a, b, c, d] = plane_model

        # 分离平面点云和非平面点云（保留原始点云，仅返回分割结果）
        inlier_cloud = _point_cloud_data.select_by_index(inliers)
        outlier_cloud = _point_cloud_data.select_by_index(inliers, invert=True)

        return {
            "message": f"平面分割完成（RANSAC）",
            "plane_model": {
                "equation": f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0",
                "normal_vector": [a, b, c],
                "distance_threshold": distance_threshold,
                "max_iterations": max_iterations
            },
            "statistics": {
                "inlier_points": len(inliers),
                "outlier_points": len(np.asarray(_point_cloud_data.points)) - len(inliers),
                "inlier_ratio": round(len(inliers)/len(np.asarray(_point_cloud_data.points)) * 100, 2)
            }
        }

    except Exception as e:
        return {"error": f"平面分割失败: {str(e)}"}


def euclidean_clustering(
    tolerance: float,
    min_cluster_size: int,
    max_cluster_size: int
) -> dict:
    """
    欧式聚类（基于 DBSCAN，分割不同物体）
    :param tolerance: 聚类容差（距离阈值，建议 0.05-0.2）
    :param min_cluster_size: 最小簇大小（建议 100-500）
    :param max_cluster_size: 最大簇大小（建议 10000-50000）
    :return: 聚类结果
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    if tolerance <= 0:
        return {"error": "聚类容差必须大于 0"}
    if min_cluster_size < 1:
        return {"error": "最小簇大小必须大于 0"}
    if max_cluster_size < min_cluster_size:
        return {"error": "最大簇大小必须大于等于最小簇大小"}

    try:
        # 构建 KD 树（加速近邻搜索）
        pcd_tree = o3d.geometry.KDTreeFlann(_point_cloud_data)
        # 执行 DBSCAN 聚类
        labels = np.array(_point_cloud_data.cluster_dbscan(
            eps=tolerance,
            min_points=min_cluster_size,
            print_progress=False
        ))

        # 分析聚类结果
        max_label = labels.max()
        clusters = []
        for i in range(max_label + 1):
            cluster_points = np.where(labels == i)[0]
            cluster_size = len(cluster_points)
            # 过滤簇大小
            if min_cluster_size <= cluster_size <= max_cluster_size:
                clusters.append({
                    "cluster_id": i,
                    "size": cluster_size,
                    "points_indices": cluster_points[:10].tolist()  # 仅返回前10个索引，避免数据过大
                })

        # 统计离群点（标签为 -1）
        outliers = np.where(labels == -1)[0]

        return {
            "message": f"欧式聚类完成（DBSCAN）",
            "parameters": {
                "tolerance": tolerance,
                "min_cluster_size": min_cluster_size,
                "max_cluster_size": max_cluster_size
            },
            "statistics": {
                "total_clusters": len(clusters),
                "total_cluster_points": sum([c["size"] for c in clusters]),
                "outlier_points": len(outliers),
                "largest_cluster_size": max([c["size"] for c in clusters]) if clusters else 0,
                "smallest_cluster_size": min([c["size"] for c in clusters]) if clusters else 0
            },
            "clusters": clusters[:10]  # 仅返回前10个簇，避免 JSON 过大
        }

    except Exception as e:
        return {"error": f"欧式聚类失败: {str(e)}"}


def save_point_cloud(file_path: str) -> dict:
    """
    保存点云到文件（支持 PCD/PLY 格式）
    :param file_path: 输出文件路径
    :return: 保存结果
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    if not file_path:
        return {"error": "文件路径不能为空"}

    try:
        file_ext = file_path.split(".")[-1].lower()
        if file_ext not in ["pcd", "ply"]:
            return {"error": f"不支持的保存格式: {file_ext}，仅支持 PCD/PLY"}

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存点云
        success = o3d.io.write_point_cloud(file_path, _point_cloud_data)
        if not success:
            return {"error": "保存失败，未知原因"}

        # 获取文件大小
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB

        return {
            "message": f"点云保存成功: {file_path}",
            "info": {
                "file_path": file_path,
                "format": file_ext.upper(),
                "file_size_mb": round(file_size, 2),
                "num_points": len(np.asarray(_point_cloud_data.points))
            }
        }

    except Exception as e:
        return {"error": f"保存失败: {str(e)}"}


def visualize() -> dict:
    """
    可视化点云（打开交互式窗口）
    :return: 可视化结果
    """
    global _point_cloud_data, _point_cloud_info

    if not _point_cloud_info.get("loaded"):
        return {"error": "未加载点云数据，请先调用 load_point_cloud"}

    if _point_cloud_data is None:
        return {"error": "点云数据为空"}

    try:
        # 创建可视化窗口（非阻塞模式，支持交互）
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Lidar AI Studio - 点云可视化", width=1280, height=720)
        vis.add_geometry(_point_cloud_data)
        
        # 设置背景色和渲染参数
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.05])  # 深灰色背景
        opt.point_size = 2.0  # 点大小
        if _point_cloud_data.has_normals():
            opt.show_normal = True  # 显示法线（如果有）

        # 运行可视化（阻塞，直到窗口关闭）
        vis.run()
        vis.destroy_window()

        return {
            "message": "可视化窗口已关闭",
            "note": "支持鼠标交互：左键旋转，右键平移，滚轮缩放"
        }

    except Exception as e:
        return {"error": f"可视化失败: {str(e)}"}


# 工具注册表（AI 调度层可见的工具定义）
TOOLS = {
    "load_point_cloud": {
        "func": load_point_cloud,
        "description": "加载点云文件（支持 PCD、PLY、LAS 格式，LAS 需要安装 laspy）",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "点云文件路径（绝对路径/相对路径）"
                }
            },
            "required": ["file_path"]
        }
    },
    "get_point_cloud_info": {
        "func": get_point_cloud_info,
        "description": "获取点云完整信息（点数、边界框、密度、法线/颜色状态、统计特征等）",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "downsample": {
        "func": downsample,
        "description": "体素网格降采样（减少点数，保留整体结构，voxel_size 建议 0.01-0.1）",
        "parameters": {
            "type": "object",
            "properties": {
                "voxel_size": {
                    "type": "number",
                    "description": "体素网格大小（越大降采样越明显）"
                }
            },
            "required": ["voxel_size"]
        }
    },
    "estimate_normals": {
        "func": estimate_normals,
        "description": "法线估计（基于 K 近邻，k_neighbors 建议 10-30）",
        "parameters": {
            "type": "object",
            "properties": {
                "k_neighbors": {
                    "type": "integer",
                    "description": "邻域点数（至少 3）"
                }
            },
            "required": ["k_neighbors"]
        }
    },
    "remove_outliers": {
        "func": remove_outliers,
        "description": "统计离群点移除（基于邻域距离标准差，nb_neighbors 建议 20-50，std_ratio 建议 1.0-2.0）",
        "parameters": {
            "type": "object",
            "properties": {
                "nb_neighbors": {
                    "type": "integer",
                    "description": "邻域点数（至少 1）"
                },
                "std_ratio": {
                    "type": "number",
                    "description": "标准差比率（大于 0）"
                }
            },
            "required": ["nb_neighbors", "std_ratio"]
        }
    },
    "segment_plane": {
        "func": segment_plane,
        "description": "平面分割（RANSAC 算法，常用于地面分割，distance_threshold 建议 0.01-0.1，max_iterations 建议 1000-10000）",
        "parameters": {
            "type": "object",
            "properties": {
                "distance_threshold": {
                    "type": "number",
                    "description": "距离阈值（点到平面的最大距离）"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "最大迭代次数"
                }
            },
            "required": ["distance_threshold", "max_iterations"]
        }
    },
    "euclidean_clustering": {
        "func": euclidean_clustering,
        "description": "欧式聚类（DBSCAN，分割不同物体，tolerance 建议 0.05-0.2，min_cluster_size 建议 100-500）",
        "parameters": {
            "type": "object",
            "properties": {
                "tolerance": {
                    "type": "number",
                    "description": "聚类容差（距离阈值，大于 0）"
                },
                "min_cluster_size": {
                    "type": "integer",
                    "description": "最小簇大小（至少 1）"
                },
                "max_cluster_size": {
                    "type": "integer",
                    "description": "最大簇大小（大于等于最小簇大小）"
                }
            },
            "required": ["tolerance", "min_cluster_size", "max_cluster_size"]
        }
    },
    "save_point_cloud": {
        "func": save_point_cloud,
        "description": "保存点云到文件（支持 PCD/PLY 格式，覆盖已有文件）",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "输出文件路径（绝对路径/相对路径）"
                }
            },
            "required": ["file_path"]
        }
    },
    "visualize": {
        "func": visualize,
        "description": "可视化点云（打开交互式窗口，支持鼠标旋转/平移/缩放，需要 GUI 环境）",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}


def handle_request(request: dict) -> dict:
    """
    处理单个 IPC 请求
    :param request: 从 Rust 接收的 JSON 请求（{"tool": "工具名", "args": {"参数": 值}}）
    :return: JSON 响应（{"result": 结果, "error": 错误信息}）
    """
    tool_name = request.get("tool")
    args = request.get("args", {})
    
    if tool_name not in TOOLS:
        return {"result": None, "error": f"未知工具：{tool_name}，可用工具：{list(TOOLS.keys())}"}
    
    try:
        func = TOOLS[tool_name]["func"]
        result = func(**args)
        return {"result": result, "error": None}
    except TypeError as e:
        return {"result": None, "error": f"参数错误：{str(e)}，请检查参数类型和必填项"}
    except Exception as e:
        return {"result": None, "error": f"工具执行失败：{str(e)}"}


def main():
    """
    主循环：从 stdin 读取 JSON 行，处理后输出 JSON 响应到 stdout
    与 Rust 端的 IPC 通信入口
    """
    # 设置 numpy 浮点精度，避免 JSON 序列化问题
    np.set_printoptions(precision=6, suppress=True)
    
    # 逐行读取 stdin（每行一个 JSON 请求）
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            # 解析请求
            request = json.loads(line)
            # 处理请求
            response = handle_request(request)
        except json.JSONDecodeError as e:
            response = {"result": None, "error": f"无效 JSON 格式：{str(e)}"}
        except Exception as e:
            response = {"result": None, "error": f"处理请求时发生意外错误：{str(e)}"}
        
        # 输出响应（强制刷新 stdout，确保 Rust 能立即接收）
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # 测试模式：如果传入 --test 参数，执行本地测试
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 示例：加载测试点云并执行一系列操作
        test_file = "test.ply"  # 替换为你的测试点云路径
        print("=== 本地测试模式 ===")
        print(f"1. 加载点云：{test_file}")
        print(json.dumps(load_point_cloud(test_file), indent=2, ensure_ascii=False))
        
        print("\n2. 获取点云信息")
        print(json.dumps(get_point_cloud_info(), indent=2, ensure_ascii=False))
        
        print("\n3. 降采样（voxel_size=0.05）")
        print(json.dumps(downsample(0.05), indent=2, ensure_ascii=False))
        
        print("\n4. 法线估计（k_neighbors=20）")
        print(json.dumps(estimate_normals(20), indent=2, ensure_ascii=False))
        
        print("\n5. 保存点云（test_output.pcd）")
        print(json.dumps(save_point_cloud("test_output.pcd"), indent=2, ensure_ascii=False))
    else:
        # IPC 通信模式
        main()