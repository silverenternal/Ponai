#!/usr/bin/env python3
"""
创建简化的 PointPillars 风格 3D 检测 ONNX 模型
使用 1D 卷积处理点云
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def create_pointpillars_onnx(output_path: str):
    """创建简化的 PointPillars ONNX 模型"""
    
    # ============ 定义输入 (固定形状用于演示) ============
    batch_size = 1
    num_points = 4096
    num_features = 4
    
    points_input = helper.make_tensor_value_info(
        'points',
        TensorProto.FLOAT,
        [batch_size, num_points, num_features]
    )
    
    # ============ 定义输出 ============
    max_det = 100
    boxes_output = helper.make_tensor_value_info(
        'boxes',
        TensorProto.FLOAT,
        [batch_size, max_det, 7]
    )
    
    scores_output = helper.make_tensor_value_info(
        'scores',
        TensorProto.FLOAT,
        [batch_size, max_det]
    )
    
    labels_output = helper.make_tensor_value_info(
        'labels',
        TensorProto.INT64,
        [batch_size, max_det]
    )
    
    # ============ 创建网络节点 ============
    nodes = []
    
    # 1. 点云归一化 - 计算均值
    nodes.append(helper.make_node(
        'ReduceMean',
        inputs=['points'],
        outputs=['points_mean'],
        axes=[1],
        keepdims=True
    ))
    
    # 2. 去中心化
    nodes.append(helper.make_node(
        'Sub',
        inputs=['points', 'points_mean'],
        outputs=['points_centered'],
    ))
    
    # 3. 创建权重 (使用 Conv1D 模拟特征提取)
    # Conv1: 4 -> 64 channels, kernel=1
    np.random.seed(42)
    conv1_weight = np.random.randn(64, 4, 1).astype(np.float32) * 0.1
    conv1_bias = np.zeros(64, dtype=np.float32)
    
    # Conv2: 64 -> 64 channels
    conv2_weight = np.random.randn(64, 64, 1).astype(np.float32) * 0.1
    conv2_bias = np.zeros(64, dtype=np.float32)
    
    # 检测头：64 -> 11 (7 box + 1 score + 3 class)
    det_head_weight = np.random.randn(11, 64, 1).astype(np.float32) * 0.01
    det_head_bias = np.zeros(11, dtype=np.float32)
    
    weights = [
        numpy_helper.from_array(conv1_weight, 'conv1.weight'),
        numpy_helper.from_array(conv1_bias, 'conv1.bias'),
        numpy_helper.from_array(conv2_weight, 'conv2.weight'),
        numpy_helper.from_array(conv2_bias, 'conv2.bias'),
        numpy_helper.from_array(det_head_weight, 'det_head.weight'),
        numpy_helper.from_array(det_head_bias, 'det_head.bias'),
    ]
    
    # 转置输入：[B, N, 4] -> [B, 4, N] 用于 Conv1D
    nodes.append(helper.make_node(
        'Transpose',
        inputs=['points_centered'],
        outputs=['points_transposed'],
        perm=[0, 2, 1]
    ))
    
    # 4. Conv1 + ReLU
    nodes.append(helper.make_node(
        'Conv',
        inputs=['points_transposed', 'conv1.weight', 'conv1.bias'],
        outputs=['conv1_out'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    nodes.append(helper.make_node(
        'Relu',
        inputs=['conv1_out'],
        outputs=['conv1_relu'],
    ))
    
    # 5. Global Max Pooling [B, 64, N] -> [B, 64]
    nodes.append(helper.make_node(
        'ReduceMax',
        inputs=['conv1_relu'],
        outputs=['global_feat'],
        axes=[2],
        keepdims=False
    ))
    
    # 6. 添加维度用于 Conv1D [B, 64] -> [B, 64, 1]
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['global_feat'],
        outputs=['global_feat_unsq'],
        axes=[2]
    ))
    
    # 7. Conv2 + ReLU
    nodes.append(helper.make_node(
        'Conv',
        inputs=['global_feat_unsq', 'conv2.weight', 'conv2.bias'],
        outputs=['conv2_out'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    nodes.append(helper.make_node(
        'Relu',
        inputs=['conv2_out'],
        outputs=['conv2_relu'],
    ))
    
    # 8. 检测头 Conv [B, 64, 1] -> [B, 11, 1]
    nodes.append(helper.make_node(
        'Conv',
        inputs=['conv2_relu', 'det_head.weight', 'det_head.bias'],
        outputs=['det_out'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    # 移除最后的维度 [B, 11, 1] -> [B, 11]
    nodes.append(helper.make_node(
        'Squeeze',
        inputs=['det_out'],
        outputs=['det_squeezed'],
        axes=[2]
    ))
    
    # 9. Sigmoid 激活
    nodes.append(helper.make_node(
        'Sigmoid',
        inputs=['det_squeezed'],
        outputs=['det_sigmoid'],
    ))
    
    # 10. Split 输出
    nodes.append(helper.make_node(
        'Split',
        inputs=['det_sigmoid'],
        outputs=['boxes_raw', 'score_raw', 'label_probs'],
        axis=-1,
        split=[7, 1, 3]
    ))
    
    # 11. 分数输出 [B, 1] -> [B]
    nodes.append(helper.make_node(
        'Squeeze',
        inputs=['score_raw'],
        outputs=['scores'],
        axes=[1]
    ))
    
    # 12. ArgMax 获取类别
    nodes.append(helper.make_node(
        'ArgMax',
        inputs=['label_probs'],
        outputs=['labels'],
        axis=-1,
    ))
    
    # 13. 边界框缩放
    box_scale = helper.make_tensor(
        'box_scale',
        TensorProto.FLOAT,
        [1, 7],
        [20.0, 20.0, 10.0, 8.0, 4.0, 4.0, 6.28]
    )
    box_offset = helper.make_tensor(
        'box_offset',
        TensorProto.FLOAT,
        [1, 7],
        [-10.0, -10.0, -5.0, -4.0, -2.0, -2.0, -3.14]
    )
    
    nodes.append(helper.make_node(
        'Mul',
        inputs=['boxes_raw', 'box_scale'],
        outputs=['boxes_scaled'],
    ))
    
    nodes.append(helper.make_node(
        'Add',
        inputs=['boxes_scaled', 'box_offset'],
        outputs=['boxes'],
    ))
    
    # ============ 创建图 ============
    graph = helper.make_graph(
        nodes,
        'PointPillars_Simplified',
        [points_input],
        [boxes_output, scores_output, labels_output],
        weights + [box_scale, box_offset]
    )
    
    # ============ 创建模型 ============
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 11)],
        producer_name='Lidar AI Studio',
        producer_version='1.0.0',
    )
    model.doc_string = 'Simplified PointPillars for 3D Object Detection'
    
    # ============ 验证模型 ============
    onnx.checker.check_model(model)
    
    # ============ 保存模型 ============
    onnx.save(model, output_path)
    print(f"✓ ONNX 模型已保存：{output_path}")
    
    param_count = sum(np.prod(w.dims) for w in weights)
    print(f"\n模型信息:")
    print(f"  输入：points [1, 4096, 4]")
    print(f"  输出：boxes [1, 100, 7]")
    print(f"  输出：scores [1, 100]")
    print(f"  输出：labels [1, 100]")
    print(f"  参数量：{param_count/1000:.1f}K")
    
    return model


def test_model(model_path: str):
    """测试 ONNX 模型推理"""
    import onnxruntime as ort
    
    session = ort.InferenceSession(model_path)
    
    # 创建模拟输入
    batch_size = 1
    num_points = 4096
    points = np.random.randn(batch_size, num_points, 4).astype(np.float32) * 10
    
    # 推理
    outputs = session.run(None, {'points': points})
    boxes, scores, labels = outputs
    
    print(f"\n推理测试:")
    print(f"  输入形状：{points.shape}")
    print(f"  boxes 输出：{boxes.shape}")
    print(f"  scores 输出：{scores.shape}")
    print(f"  labels 输出：{labels.shape}")

    # 显示检测结果 (当前模型输出单个检测)
    print(f"\n  检测结果:")
    label_names = ['Car', 'Pedestrian', 'Cyclist']
    label_id = int(labels.flatten()[0])
    label = label_names[label_id] if label_id < 3 else f'Class{label_id}'
    print(f"    {label}: score={float(scores[0]):.3f}, box={boxes.flatten().round(2)}")

    return boxes, scores, labels


def benchmark_model(model_path: str, num_runs: int = 100):
    """性能测试"""
    import onnxruntime as ort
    import time
    
    session = ort.InferenceSession(model_path)
    
    # 预热
    for _ in range(10):
        points = np.random.randn(1, 4096, 4).astype(np.float32)
        session.run(None, {'points': points})
    
    # 正式测试
    latencies = []
    for _ in range(num_runs):
        points = np.random.randn(1, 4096, 4).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {'points': points})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    print(f"\nCPU 性能测试 ({num_runs} 次运行):")
    print(f"  平均延迟：{avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  模型大小：{os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    return avg_latency, fps


def main():
    """主程序"""
    print("="*60)
    print("创建简化的 PointPillars ONNX 模型")
    print("="*60)
    
    os.makedirs('python_tools/models', exist_ok=True)
    model_path = 'python_tools/models/pointpillars.onnx'
    
    # 创建模型
    model = create_pointpillars_onnx(model_path)
    
    # 测试推理
    print("\n" + "="*60)
    print("测试模型推理")
    print("="*60)
    test_model(model_path)
    
    # 性能测试
    print("\n" + "="*60)
    print("CPU 性能基准测试")
    print("="*60)
    benchmark_model(model_path)
    
    print("\n" + "="*60)
    print("✓ 模型创建和测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
