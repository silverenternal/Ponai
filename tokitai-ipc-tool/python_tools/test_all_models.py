#!/usr/bin/env python3
"""
Test all PointPillars models - ONNX inference benchmark.

This script tests all available PointPillars models:
1. pointpillars.onnx (simplified demo, 22KB)
2. pointpillars_realistic.onnx (authentic architecture, 192KB)
3. pointpillars_simple.onnx (PyTorch export, 22MB)
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

models_dir = Path(__file__).parent / 'models'

# Available ONNX models
models = {
    'pointpillars.onnx': 'Simplified demo (5.2K params)',
    'pointpillars_realistic.onnx': 'Authentic architecture (48.4K params)',
    'pointpillars_simple.onnx': 'PyTorch export (5.6M params)',
}

print("=" * 70)
print("PointPillars ONNX Model Benchmark")
print("=" * 70)

# Generate synthetic point cloud
def generate_pointcloud(num_points=30000):
    x = np.random.uniform(0, 70, num_points).astype(np.float32)
    y = np.random.uniform(-40, 40, num_points).astype(np.float32)
    z = np.random.uniform(-3, 2, num_points).astype(np.float32)
    intensity = np.random.uniform(0, 1, num_points).astype(np.float32)
    return np.stack([x, y, z, intensity], axis=1)

# Test each model
for model_name, description in models.items():
    model_path = models_dir / model_name
    
    if not model_path.exists():
        print(f"\n❌ {model_name}: Not found")
        continue
    
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name}")
    print(f"  Description: {description}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    try:
        # Load model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_path), sess_options, providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        
        print(f"  Input: {input_name}")
        print(f"  Outputs: {output_names}")
        
        # Get expected input shape
        input_shape = session.get_inputs()[0].shape
        expected_points = input_shape[1] if len(input_shape) > 1 else None
        
        # Warm up (add batch dimension if model expects 3D input)
        num_pts = expected_points if expected_points else 10000
        dummy_input = generate_pointcloud(num_pts)
        if len(input_shape) == 3:
            dummy_input = dummy_input[np.newaxis, ...]
        _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        num_tests = 100
        latencies = []
        
        for _ in range(num_tests):
            pc = generate_pointcloud(num_pts)
            if len(input_shape) == 3:
                pc = pc[np.newaxis, ...]
            
            start = time.perf_counter()
            outputs = session.run(None, {input_name: pc})
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        latencies = np.array(latencies)
        fps = 1000 / latencies.mean()
        
        print(f"\n  Performance (CPU):")
        print(f"    Mean latency: {latencies.mean():.2f} ms")
        print(f"    Std latency:  {latencies.std():.2f} ms")
        print(f"    FPS:          {fps:.1f}")
        print(f"  ✓ Model loaded and running successfully")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n{'=' * 70}")
print("Benchmark complete!")
print(f"{'=' * 70}")

# Summary
print("\nSummary:")
print("  - pointpillars.onnx: Fastest, good for framework testing")
print("  - pointpillars_realistic.onnx: Best balance of speed and realism")
print("  - pointpillars_simple.onnx: Full-featured but slower (unoptimized)")
print("\nNote: All ONNX models have random weights (not pretrained)")
print("      For real detections, use the PyTorch model:")
print("      - pointpillars_kitti_3class.pth (18.51 MB, official MMDetection3D weights)")
