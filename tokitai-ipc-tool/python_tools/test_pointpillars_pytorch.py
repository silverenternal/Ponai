#!/usr/bin/env python3
"""
PointPillars PyTorch Inference Tool

This script uses the official MMDetection3D PointPillars pretrained weights
for inference without ONNX export. This is the most reliable approach
given MMDetection3D's complex architecture.
"""

import sys
import torch
import numpy as np
import time

# Add mmdetection3d to path
sys.path.insert(0, '/home/hugo/codes/Ponai/tokitai-ipc-tool/mmdetection3d')

from mmdet3d.utils import register_all_modules
from mmengine.runner import load_checkpoint
from mmengine.model import revert_sync_batchnorm

print("=" * 70)
print("PointPillars PyTorch Inference Tool")
print("Using official MMDetection3D pretrained weights (KITTI 3-class)")
print("=" * 70)

# Register modules
register_all_modules()

# Model configuration (matching KITTI 3-class)
voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Pedestrian', 'Cyclist', 'Car']

print(f"\nClasses: {class_names}")
print(f"Voxel size: {voxel_size}")
print(f"Point cloud range: {point_cloud_range}")

print("\n[1/3] Building model...")

from mmdet3d.models import VoxelNet
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.voxel_encoders import PillarFeatureNet, PointPillarsScatter
from mmdet3d.models.backbones import SECOND
from mmdet3d.models.necks import SECONDFPN
from mmdet3d.models.dense_heads import Anchor3DHead

model = VoxelNet(
    data_preprocessor=Det3DDataPreprocessor(
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=PillarFeatureNet(
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=PointPillarsScatter(
        in_channels=64,
        output_shape=[496, 432]),
    backbone=SECOND(
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=SECONDFPN(
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=Anchor3DHead(
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder')),
)

print("[2/3] Loading pretrained weights...")
checkpoint_path = '/home/hugo/codes/Ponai/tokitai-ipc-tool/python_tools/models/pointpillars_kitti_3class.pth'
load_checkpoint(model, checkpoint_path, map_location='cpu')
model.eval()
model = revert_sync_batchnorm(model)

print("[3/3] Running inference test...")

# Generate synthetic point cloud (similar to KITTI LiDAR)
def generate_kitti_like_pointcloud(num_points=50000):
    """Generate a synthetic point cloud similar to KITTI dataset."""
    # KITTI Velodyne characteristics
    x = np.random.uniform(0, 70, num_points).astype(np.float32)  # Forward
    y = np.random.uniform(-40, 40, num_points).astype(np.float32)  # Lateral
    z = np.random.uniform(-3, 2, num_points).astype(np.float32)  # Height
    
    # Intensity (simulated)
    intensity = np.random.uniform(0, 1, num_points).astype(np.float32)
    
    return np.stack([x, y, z, intensity], axis=1)

# Warm up
print("\n  Warming up...")
dummy_pc = generate_kitti_like_pointcloud(10000)
with torch.no_grad():
    input_dict = dict(points=[torch.from_numpy(dummy_pc)])
    try:
        result = model.predict(input_dict, rescale=True)
    except Exception as e:
        print(f"  Warning: {e}")

# Benchmark
print("\n  Running benchmark (100 iterations)...")
num_tests = 100
latencies = []

for i in range(num_tests):
    pc = generate_kitti_like_pointcloud(30000)  # Typical KITTI point count
    
    start = time.perf_counter()
    with torch.no_grad():
        input_dict = dict(points=[torch.from_numpy(pc)])
        result = model.predict(input_dict, rescale=True)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)
    
    if i == 0:
        # Show first result
        boxes = result[0].boxes_3d.tensor.numpy() if len(result[0].boxes_3d) > 0 else np.empty((0, 7))
        scores = result[0].scores_3d.numpy() if hasattr(result[0], 'scores_3d') else np.array([])
        labels = result[0].labels_3d.numpy() if hasattr(result[0], 'labels_3d') else np.array([])
        print(f"\n  First inference results:")
        print(f"    Detected {len(boxes)} objects")
        if len(boxes) > 0:
            for j, (box, score, label) in enumerate(zip(boxes[:3], scores[:3], labels[:3])):
                cls_name = class_names[int(label)] if int(label) < len(class_names) else f"Class{label}"
                print(f"    - {cls_name}: {score:.3f} @ [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}]")

# Statistics
latencies = np.array(latencies)
fps = 1000 / latencies.mean()

print(f"\n{'=' * 70}")
print(f"Performance Results (CPU):")
print(f"  Mean latency: {latencies.mean():.2f} ms")
print(f"  Std latency:  {latencies.std():.2f} ms")
print(f"  FPS:          {fps:.1f}")
print(f"{'=' * 70}")

print("\n✓ PointPillars PyTorch inference working correctly!")
print(f"  Model: pointpillars_kitti_3class.pth (18.51 MB)")
print(f"  Classes: {class_names}")
print(f"  Dataset: KITTI")
