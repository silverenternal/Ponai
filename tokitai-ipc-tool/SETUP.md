# Lidar AI Studio - 环境配置指南

## 当前环境状态

| 组件 | 状态 | 版本 |
|------|------|------|
| Rust | ✅ 已安装 | 1.94.0 |
| Python | ✅ 已安装 | 3.14.3 |
| 虚拟环境 | ✅ 已创建 | `.venv/` |
| numpy | ✅ 已安装 | 2.4.4 |
| onnxruntime | ✅ 已安装 | 1.24.4 |
| CMake | ❌ 待安装 | - |
| jsoncpp | ❌ 待安装 | - |
| Open3D | ⚠️ 需手动安装 | - |

---

## 配置步骤

### 步骤 1: 安装系统级依赖（需要 sudo）

```bash
# 安装 CMake 和 jsoncpp（C++ 工具必需）
sudo pacman -S cmake jsoncpp

# 安装 Open3D（Python 3.14 无 pip 支持，需用系统包）
# 如果 AUR 可用：
yay -S python-open3d

# 或从 AUR 手动编译：
git clone https://aur.archlinux.org/python-open3d.git
cd python-open3d
makepkg -si
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

# 验证 Python 环境
python --version
pip list

# 验证 ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.__version__)"

# 验证 numpy
python -c "import numpy; print(numpy.__version__)"
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
```

---

## Open3D 安装问题

**问题**: Python 3.14 太新，Open3D 官方 wheel 包暂不支持。

**解决方案**:

### 方案 A: 使用 AUR 安装（推荐）

```bash
yay -S python-open3d
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
```

### 方案 C: 跳过 Open3D（仅使用 ONNX 推理）

如果只需要 ONNX 推理功能，可以暂时不安装 Open3D。

---

## 已安装的 Python 包

```
numpy       - 数值计算
onnxruntime - ONNX 推理引擎
```

---

## 可选：Ollama AI 服务

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动服务
ollama serve

# 拉取模型
ollama pull llama3.2
```

---

## 快速启动

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行 Rust 项目
cargo run
```
