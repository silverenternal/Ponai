#!/usr/bin/env python3
"""
Export MMDetection3D PointPillars model to ONNX.

This script loads the pretrained PointPillars model and exports it to ONNX format.
Due to MMDetection3D's complex architecture, we export a simplified inference wrapper.
"""

import sys
import torch
import numpy as np

# Add mmdetection3d to path
sys.path.insert(0, '/home/hugo/codes/Ponai/tokitai-ipc-tool/mmdetection3d')

from mmdet3d.utils import register_all_modules
from mmengine.runner import load_checkpoint
from mmengine.model import revert_sync_batchnorm

print("=" * 60)
print("PointPillars ONNX Export Tool")
print("=" * 60)

# Register all modules
register_all_modules()

# Model config (simplified)
voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

print("\n[1/4] Building model architecture...")

# Import model components
from mmdet3d.models import VoxelNet
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.voxel_encoders import PillarFeatureNet, PointPillarsScatter
from mmdet3d.models.backbones import SECOND
from mmdet3d.models.necks import SECONDFPN
from mmdet3d.models.dense_heads import Anchor3DHead

# Build model
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
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
)

print("[2/4] Loading pretrained weights...")
checkpoint_path = '/home/hugo/codes/Ponai/tokitai-ipc-tool/python_tools/models/pointpillars_kitti_3class.pth'
load_checkpoint(model, checkpoint_path, map_location='cpu')
model.eval()
model = revert_sync_batchnorm(model)

print("[3/4] Creating inference wrapper...")

class PointPillarsWrapper(torch.nn.Module):
    """Wrapper for PointPillars to simplify ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, points):
        """
        Args:
            points: Tensor of shape (N, 4) - (x, y, z, intensity)
        Returns:
            boxes: (N_det, 7) - (x, y, z, w, l, h, yaw)
            scores: (N_det,) - confidence scores
            labels: (N_det,) - class labels (0=Pedestrian, 1=Cyclist, 2=Car)
        """
        # Wrap points in list as expected by model
        input_dict = dict(points=[points])
        
        # Run inference
        result = self.model.predict(input_dict, rescale=True)
        
        # Extract results
        boxes_3d = result[0].boxes_3d.tensor  # (N, 7)
        scores_3d = result[0].scores_3d  # (N,)
        labels_3d = result[0].labels_3d  # (N,)
        
        return boxes_3d, scores_3d, labels_3d

wrapper = PointPillarsWrapper(model)

print("[4/4] Exporting to ONNX...")

# Create dummy input
num_points = 20000
dummy_points = torch.randn(num_points, 4, dtype=torch.float32)
dummy_points[:, :3] *= 30  # Scale to realistic range

output_path = '/home/hugo/codes/Ponai/tokitai-ipc-tool/python_tools/models/pointpillars_kitti_3class.onnx'

try:
    torch.onnx.export(
        wrapper,
        dummy_points,
        output_path,
        input_names=['points'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={
            'points': {0: 'num_points'},
            'boxes': {0: 'num_detections'},
            'scores': {0: 'num_detections'},
            'labels': {0: 'num_detections'}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    print(f"\n✓ Successfully exported to: {output_path}")
    
    # Check file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
    
except Exception as e:
    print(f"\n✗ Export failed: {e}")
    print("\nNote: MMDetection3D models have complex control flow that may not")
    print("export cleanly to ONNX. Consider using PyTorch inference directly.")
    
    # Fallback: Save as TorchScript
    print("\nAttempting TorchScript export as fallback...")
    try:
        scripted = torch.jit.trace(wrapper, dummy_points)
        ts_path = output_path.replace('.onnx', '.pt')
        scripted.save(ts_path)
        print(f"✓ TorchScript model saved to: {ts_path}")
    except Exception as e2:
        print(f"✗ TorchScript export also failed: {e2}")
