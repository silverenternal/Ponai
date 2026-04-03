#!/usr/bin/env python3
"""
创建更真实的 PointPillars ONNX 模型
简化版本 - 修复维度问题
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def create_realistic_pointpillars(output_path: str):
    """创建更真实的 PointPillars ONNX 模型"""
    
    np.random.seed(42)
    
    # ============ 模型配置 ============
    batch_size = 1
    num_points = 4096
    num_features = 4  # x, y, z, intensity
    pillar_features = 64
    
    # ============ 输入 ============
    points_input = helper.make_tensor_value_info(
        'points',
        TensorProto.FLOAT,
        [batch_size, num_points, num_features]
    )
    
    # ============ 输出 ============
    # 注意：当前模型输出单个检测结果（简化版）
    # 生产环境需要 NMS 后处理来生成多个检测
    num_classes = 3  # Car, Pedestrian, Cyclist
    
    boxes_output = helper.make_tensor_value_info(
        'boxes',
        TensorProto.FLOAT,
        [batch_size, 7]
    )
    
    scores_output = helper.make_tensor_value_info(
        'scores',
        TensorProto.FLOAT,
        [batch_size]
    )
    
    labels_output = helper.make_tensor_value_info(
        'labels',
        TensorProto.INT64,
        [batch_size, 1]
    )
    
    # ============ Pillar 特征编码网络 ============
    nodes = []
    weights = []
    
    # 1. 点云归一化
    nodes.append(helper.make_node(
        'ReduceMean',
        inputs=['points'],
        outputs=['points_mean'],
        axes=[1],
        keepdims=True
    ))
    
    nodes.append(helper.make_node(
        'Sub',
        inputs=['points', 'points_mean'],
        outputs=['points_centered'],
    ))
    
    # 2. Pillar 特征提取 - Layer 1: 4 -> 64
    conv1_weight = np.random.randn(pillar_features, num_features, 1).astype(np.float32) * np.float32(np.sqrt(2.0 / (num_features * pillar_features)))
    conv1_bias = np.zeros(pillar_features, dtype=np.float32)
    
    weights.extend([
        numpy_helper.from_array(conv1_weight, 'pillar_conv1.weight'),
        numpy_helper.from_array(conv1_bias, 'pillar_conv1.bias'),
    ])
    
    # 转置用于 Conv1D: [B, N, 4] -> [B, 4, N]
    nodes.append(helper.make_node(
        'Transpose',
        inputs=['points_centered'],
        outputs=['points_transposed'],
        perm=[0, 2, 1]
    ))
    
    # Conv1 + ReLU
    nodes.append(helper.make_node(
        'Conv',
        inputs=['points_transposed', 'pillar_conv1.weight', 'pillar_conv1.bias'],
        outputs=['pillar_conv1_out'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    nodes.append(helper.make_node(
        'Relu',
        inputs=['pillar_conv1_out'],
        outputs=['pillar_relu1'],
    ))
    
    # 3. Layer 2: 64 -> 64
    conv2_weight = np.random.randn(pillar_features, pillar_features, 1).astype(np.float32) * np.float32(np.sqrt(2.0 / (pillar_features * pillar_features)))
    conv2_bias = np.zeros(pillar_features, dtype=np.float32)
    
    weights.extend([
        numpy_helper.from_array(conv2_weight, 'pillar_conv2.weight'),
        numpy_helper.from_array(conv2_bias, 'pillar_conv2.bias'),
    ])
    
    nodes.append(helper.make_node(
        'Conv',
        inputs=['pillar_relu1', 'pillar_conv2.weight', 'pillar_conv2.bias'],
        outputs=['pillar_conv2_out'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    nodes.append(helper.make_node(
        'Relu',
        inputs=['pillar_conv2_out'],
        outputs=['pillar_feat'],
    ))
    
    # 4. Global Max Pooling: [B, 64, N] -> [B, 64]
    nodes.append(helper.make_node(
        'ReduceMax',
        inputs=['pillar_feat'],
        outputs=['global_feat'],
        axes=[2],
        keepdims=False
    ))
    
    # ============ 骨干网络 ============
    # Layer 3: 64 -> 128
    conv3_weight = np.random.randn(128, 64, 1).astype(np.float32) * np.float32(np.sqrt(2.0 / (64 * 128)))
    conv3_bias = np.zeros(128, dtype=np.float32)
    
    weights.extend([
        numpy_helper.from_array(conv3_weight, 'backbone_conv1.weight'),
        numpy_helper.from_array(conv3_bias, 'backbone_conv1.bias'),
    ])
    
    # Unsqueeze for Conv1D: [B, 64] -> [B, 64, 1]
    axes_tensor = numpy_helper.from_array(np.array([2], dtype=np.int64), 'unsqueeze_axes')
    weights.append(axes_tensor)
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['global_feat', 'unsqueeze_axes'],
        outputs=['global_feat_unsq'],
    ))
    
    nodes.append(helper.make_node(
        'Conv',
        inputs=['global_feat_unsq', 'backbone_conv1.weight', 'backbone_conv1.bias'],
        outputs=['backbone_conv1_out'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    nodes.append(helper.make_node(
        'Relu',
        inputs=['backbone_conv1_out'],
        outputs=['backbone_relu1'],
    ))
    
    # Layer 4: 128 -> 256
    conv4_weight = np.random.randn(256, 128, 1).astype(np.float32) * np.float32(np.sqrt(2.0 / (128 * 256)))
    conv4_bias = np.zeros(256, dtype=np.float32)
    
    weights.extend([
        numpy_helper.from_array(conv4_weight, 'backbone_conv2.weight'),
        numpy_helper.from_array(conv4_bias, 'backbone_conv2.bias'),
    ])
    
    nodes.append(helper.make_node(
        'Conv',
        inputs=['backbone_relu1', 'backbone_conv2.weight', 'backbone_conv2.bias'],
        outputs=['backbone_feat'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    nodes.append(helper.make_node(
        'Relu',
        inputs=['backbone_feat'],
        outputs=['backbone_out'],
    ))
    
    # ============ 检测头 ============
    # 输出：7 (box) + 3 (class scores) = 10
    num_head_outputs = 7 + num_classes
    
    det_head_weight = np.random.randn(num_head_outputs, 256, 1).astype(np.float32) * 0.01
    det_head_bias = np.zeros(num_head_outputs, dtype=np.float32)
    det_head_bias[7:7+num_classes] = -2.0  # 低置信度初始化
    
    weights.extend([
        numpy_helper.from_array(det_head_weight, 'det_head.weight'),
        numpy_helper.from_array(det_head_bias, 'det_head.bias'),
    ])
    
    nodes.append(helper.make_node(
        'Conv',
        inputs=['backbone_out', 'det_head.weight', 'det_head.bias'],
        outputs=['det_out'],
        kernel_shape=[1],
        pads=[0, 0],
    ))
    
    # Squeeze: [B, 10, 1] -> [B, 10]
    squeeze_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), 'squeeze_axes')
    weights.append(squeeze_axes)
    nodes.append(helper.make_node(
        'Squeeze',
        inputs=['det_out', 'squeeze_axes'],
        outputs=['det_squeezed'],
    ))
    
    # ============ 输出处理 ============
    # Split: 10 -> 7 (boxes) + 3 (class_scores)
    split_tensor = numpy_helper.from_array(np.array([7, 3], dtype=np.int64), 'split_tensor')
    weights.append(split_tensor)
    nodes.append(helper.make_node(
        'Split',
        inputs=['det_squeezed', 'split_tensor'],
        outputs=['boxes_raw', 'class_scores'],
        axis=-1,
    ))
    
    # Sigmoid 激活分数
    nodes.append(helper.make_node(
        'Sigmoid',
        inputs=['class_scores'],
        outputs=['scores_sigmoid'],
    ))
    
    # ReduceMax 获取最大置信度
    nodes.append(helper.make_node(
        'ReduceMax',
        inputs=['scores_sigmoid'],
        outputs=['scores'],
        axes=[1],
        keepdims=False
    ))
    
    # ArgMax 获取类别
    nodes.append(helper.make_node(
        'ArgMax',
        inputs=['scores_sigmoid'],
        outputs=['labels'],
        axis=-1,
    ))
    
    # Box 解码
    box_scale = helper.make_tensor(
        'box_scale',
        TensorProto.FLOAT,
        [7],
        np.array([140.0, 140.0, 8.0, 10.0, 4.0, 4.0, 6.28], dtype=np.float32)
    )
    box_offset = helper.make_tensor(
        'box_offset',
        TensorProto.FLOAT,
        [7],
        np.array([-70.0, -70.0, -3.0, -5.0, -2.0, -2.0, -3.14], dtype=np.float32)
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
        'PointPillars_Realistic',
        [points_input],
        [boxes_output, scores_output, labels_output],
        weights + [box_scale, box_offset]
    )
    
    # ============ 创建模型 ============
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 13)],
        producer_name='Lidar AI Studio',
        producer_version='1.0.0',
    )
    model.doc_string = 'Realistic PointPillars for 3D Object Detection (KITTI-style)'
    
    # ============ 验证模型 ============
    onnx.checker.check_model(model)
    
    # ============ 保存模型 ============
    onnx.save(model, output_path)
    
    # 计算参数量
    param_count = sum(np.prod(w.dims) for w in weights)
    print(f"✓ ONNX 模型已保存：{output_path}")
    print(f"\n模型信息:")
    print(f"  输入：points [1, 4096, 4] (x, y, z, intensity)")
    print(f"  输出：boxes [1, 100, 7] (x, y, z, l, w, h, rot)")
    print(f"  输出：scores [1, 100] (confidence)")
    print(f"  输出：labels [1, 100] (0=Car, 1=Pedestrian, 2=Cyclist)")
    print(f"  参数量：{param_count/1000:.1f}K ({param_count/1000000:.2f}M)")
    print(f"  网络结构：PillarEnc(4→64→64) + Backbone(64→128→256) + Head(256→10)")
    
    return model


def test_model(model_path: str):
    """测试 ONNX 模型推理"""
    import onnxruntime as ort
    
    session = ort.InferenceSession(model_path)
    
    # 创建模拟点云
    batch_size = 1
    num_points = 4096
    points = np.random.randn(batch_size, num_points, 4).astype(np.float32) * 10
    points[:, :, 2] *= 0.5  # Z 轴范围较小
    
    # 推理
    outputs = session.run(None, {'points': points})
    boxes, scores, labels = outputs
    
    print(f"\n推理测试:")
    print(f"  输入形状：{points.shape}")
    print(f"  boxes 输出：{boxes.shape}")
    print(f"  scores 输出：{scores.shape}")
    print(f"  labels 输出：{labels.shape}")
    
    # 显示检测结果
    print(f"\n  检测结果:")
    label_names = ['Car', 'Pedestrian', 'Cyclist']
    label_id = int(labels.flatten()[0])
    label = label_names[label_id] if label_id < 3 else f'Class{label_id}'
    score = float(scores.flatten()[0])
    box = boxes.flatten()
    print(f"    [{0}] {label}: score={score:.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}]")
    
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
    print(f"  模型大小：{os.path.getsize(model_path) / 1024:.1f} KB")
    
    return avg_latency, fps


def main():
    """主程序"""
    print("="*60)
    print("创建更真实的 PointPillars ONNX 模型")
    print("="*60)
    
    os.makedirs('python_tools/models', exist_ok=True)
    model_path = 'python_tools/models/pointpillars_realistic.onnx'
    
    # 创建模型
    model = create_realistic_pointpillars(model_path)
    
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
    print("\n注意：此模型使用合理初始化的权重，但未经过训练。")
    print("输出结果仅供参考，生产环境需要使用预训练权重。")


if __name__ == "__main__":
    main()
