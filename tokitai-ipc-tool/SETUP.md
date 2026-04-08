# Lidar AI Studio - 环境配置指南

**最后更新**: 2026 年 4 月 8 日

## 当前环境状态

| 组件 | 状态 | 版本 | 说明 |
|------|------|------|------|
| Rust | ✅ 已安装 | 1.94.0 | 支持最新特性 |
| Python | ✅ 已安装 | 3.14.3 | 需注意 Open3D 兼容性 |
| 虚拟环境 | ✅ 已创建 | `.venv/` | 项目专用 |
| numpy | ✅ 已安装 | 2.4.4 | 数值计算基础 |
| onnxruntime | ✅ 已安装 | 1.24.4 | CPU 推理引擎 |
| Open3D | ⚠️ 需系统包 | - | Python 3.14 无 pip 支持 |
| CMake | ❌ 可选 | - | C++ 工具需要 |
| jsoncpp | ❌ 可选 | - | C++ 工具需要 |
| Ollama | ❌ 可选 | - | AI 调度功能需要 |

---

## 配置步骤

### 步骤 1: 安装系统级依赖（可选，需要 sudo）

**C++ 工具依赖**（如只需要 Python 工具，可跳过）:

```bash
# Arch Linux
sudo pacman -S cmake jsoncpp

# Ubuntu/Debian
sudo apt-get install cmake libjsoncpp-dev

# macOS
brew install cmake jsoncpp
```

**Open3D**（Python 3.14 需系统包）:

```bash
# Arch Linux (推荐)
sudo pacman -S python-open3d

# 或从 AUR 安装
yay -S python-open3d

# Ubuntu/Debian (Python 3.10/3.11)
sudo apt-get install python3-open3d

# macOS
brew install open3d
```

### 步骤 2: 激活 Python 虚拟环境

```bash
cd /home/hugo/codes/Ponai/tokitai-ipc-tool
source .venv/bin/activate
```

### 步骤 3: 验证环境

```bash
# 验证 Rust
cargo --version
# 输出：cargo 1.94.0

# 验证 Python 环境
python --version
# 输出：Python 3.14.3

pip list
# 应显示：numpy, onnxruntime

# 验证 ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.__version__)"
# 输出：1.24.4

# 验证 numpy
python -c "import numpy; print(numpy.__version__)"
# 输出：2.4.4

# 验证 Open3D (如果已安装)
python -c "import open3d; print(open3d.__version__)"
```

### 步骤 4: 构建 Rust 项目

```bash
cd /home/hugo/codes/Ponai/tokitai-ipc-tool
cargo build
```

### 步骤 5: 编译 C++ 工具（可选）

安装 CMake 和 jsoncpp 后：

```bash
cd /home/hugo/codes/Ponai/tokitai-ipc-tool/cpp_tools
mkdir -p build && cd build
cmake ..
make
# 输出：cpp_tools/build/pointcloud_tools_cpp
```

### 步骤 6: 启动 HTTP 服务（可选）

```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动服务
cd python_tools
./start_server.sh --host 0.0.0.0 --port 8080 --api-key your-key

# 验证服务
curl http://localhost:8080/health
```

### 步骤 7: 配置 Ollama（可选，AI 调度功能需要）

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动服务
ollama serve

# 拉取模型
ollama pull llama3.2

# 验证
curl http://localhost:11434/api/tags
```

---

## Open3D 安装问题

**问题**: Python 3.14 太新，Open3D 官方 wheel 包暂不支持。

**解决方案**:

### 方案 A: 使用系统包管理器安装（推荐）

```bash
# Arch Linux
sudo pacman -S python-open3d

# 复制 open3d 到虚拟环境
cp -r /usr/lib/python3.14/site-packages/open3d* .venv/lib/python3.14/site-packages/
```

### 方案 B: 使用 pyenv 安装 Python 3.12

```bash
# 安装 pyenv
curl https://pyenv.run | bash

# 重启终端或执行：
exec $SHELL

# 安装 Python 3.12
pyenv install 3.12.8
pyenv virtualenv 3.12.8 lidar-env

# 在项目目录激活
cd /home/hugo/codes/Ponai/tokitai-ipc-tool
pyenv local lidar-env
pip install -r requirements.txt

# 安装 Open3D
pip install open3d
```

### 方案 C: 跳过 Open3D（仅使用 ONNX 推理）

如果只需要 ONNX 推理功能，可以暂时不安装 Open3D。

---

## 已安装的 Python 包

```
numpy        - 数值计算基础
onnxruntime  - ONNX 推理引擎 (CPU)
```

---

## 快速启动

### 本地开发（IPC 模式）

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行 Rust 项目
cargo run
```

### HTTP 服务模式

```bash
# 终端 1: 启动 HTTP 服务
source .venv/bin/activate
cd python_tools
./start_server.sh --port 8080

# 终端 2: 运行 Rust 客户端
cargo run
```

### AI 调度模式（需要 Ollama）

```bash
# 终端 1: 启动 Ollama
ollama serve

# 终端 2: 运行 Rust 项目
cargo run
```
