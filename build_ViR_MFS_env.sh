#!/bin/bash

# --- 严格配置 ---
ENV_NAME="metafusion_venv"
PYTHON_VER="3.10"
PYTHON_PATH="/usr/local/bin/python3.10"
TARGET_DIR=$(pwd)

# 强制清空代理
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "------------------------------------------------"
echo "🚀 ViR_MFS 环境部署 (Torch 2.3.0 + CU118 + python 3.10)"
echo "------------------------------------------------"

# --- 智能安装函数 (优先使用官方源确保元数据最新) ---
smart_pip_install() {
    local args="$1"
    local mirrors=("https://pypi.org/simple" "https://mirrors.aliyun.com/pypi/simple/" "https://pypi.tuna.tsinghua.edu.cn/simple")
    
    for mirror in "${mirrors[@]}"; do
        echo "📡 尝试源: $mirror ..."
        if pip install $args -i "$mirror" --timeout 30 --retries 1; then
            return 0
        fi
    done
    return 1
}

# --- Python 版本检测与编译 ---
check_python_exists() {
    if [ -f "$PYTHON_PATH" ] || command -v python3.10 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

if check_python_exists; then
    [ ! -f "$PYTHON_PATH" ] && PYTHON_PATH=$(command -v python3.10)
    echo "✅ 系统已存在 Python 3.10 ($PYTHON_PATH)。"
else
    echo "🛠️ 1/7: 配置编译环境..."
    sudo apt update
    sudo apt install -y build-essential zlib1g-dev libncurses-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev
    echo "🔍 2/7: 编译 Python $PYTHON_VER..."
    PY_TGZ="Python-$PYTHON_VER.tgz"
    wget -c "https://www.python.org/ftp/python/$PYTHON_VER/$PY_TGZ"
    tar -xzf "$PY_TGZ"
    cd "Python-$PYTHON_VER"
    ./configure --enable-optimizations
    make -j$(nproc)
    sudo make altinstall
    cd ..
    rm -rf "Python-$PYTHON_VER" && rm -f "$PY_TGZ"
fi

# 3. 创建虚拟环境
echo "📦 3/7: 创建 3.10 虚拟环境..."
$PYTHON_PATH -m venv $ENV_NAME
source $ENV_NAME/bin/activate

# 4. 升级核心构建工具 (直接规避 wheel/packaging 冲突)
echo "🆙 4/7: 升级构建基础环境..."
smart_pip_install "--upgrade pip setuptools wheel packaging"

# 5. 安装 Torch 2.3.0(CU118)
echo "💿 5/7: 安装 Torch 2.3.0 和 Torchvision 0.18.0 (CU118)..."
if pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118; then
    echo "✅ Torch 安装成功。"
else
    echo "❌ Torch 安装失败，请检查网络。"
    exit 1
fi

# 6. 安装业务清单
if [ -f "requirements.txt" ]; then
    echo "📋 6/7: 正在安装业务依赖..."
    
    # 强制修复 pycocotools 孤岛编译环境
    echo "🔧 优先构建 pycocotools..."
    smart_pip_install "Cython<3.0.0 numpy<2.0.0"
    smart_pip_install "pycocotools --no-build-isolation --no-cache-dir"
    
    # 批量安装剩余的安全依赖 (带有官方源兜底)
    echo "📦 正在安装 requirements.txt..."
    if smart_pip_install "-r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118"; then
        INSTALL_STATUS=0
    else
        INSTALL_STATUS=1
    fi
else
    INSTALL_STATUS=0
fi

# 7. 最终校验
if [ $INSTALL_STATUS -eq 0 ]; then
    echo "------------------------------------------------"
    echo "🧹 部署成功！执行最终自检..."
    python -c "import torch; print('🚀 CUDA 状态:', torch.cuda.is_available()); print('📦 Torch 版本:', torch.__version__)"
    echo "------------------------------------------------"
else
    echo "⚠️ 依赖安装中途出错。"
    exit 1
fi

exec bash --rcfile <(echo "source ~/.bashrc; source $TARGET_DIR/$ENV_NAME/bin/activate; echo '🔥 环境已就绪！'")