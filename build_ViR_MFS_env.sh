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

# --- 多镜像源轮询安装函数 ---
smart_pip_install() {
    local args="$1"
    local mirrors=("https://mirrors.aliyun.com/pypi/simple/" "https://mirrors.volces.com/pypi/simple/" "https://pypi.tuna.tsinghua.edu.cn/simple")
    local hosts=("mirrors.aliyun.com" "mirrors.volces.com" "pypi.tuna.tsinghua.edu.cn")

    for i in "${!mirrors[@]}"; do
        echo "📡 尝试镜像源: ${hosts[$i]} ..."
        if pip install $args -i "${mirrors[$i]}" --trusted-host "${hosts[$i]}" --timeout 30 --retries 1; then
            return 0
        fi
        echo "⚠️ 镜像源 ${hosts[$i]} 连接失败，尝试下一个..."
    done
    return 1
}

# --- Python 版本检测 ---
check_python_exists() {
    if [ -f "$PYTHON_PATH" ] || command -v python3.10 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# 1-2 步：完美版源码编译
if check_python_exists; then
    [ ! -f "$PYTHON_PATH" ] && PYTHON_PATH=$(command -v python3.10)
    echo "✅ 系统已存在 Python 3.10 ($PYTHON_PATH)，跳过编译阶段。"
else
    echo "🛠️ 1/7: 正在配置系统编译环境 (补齐所有底层 C 库)..."
    sudo apt update
    # 加入了 liblzma-dev，彻底解决各种底层报错
    sudo apt install -y build-essential zlib1g-dev libncurses-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev

    echo "🔍 2/7: 正在获取并编译 Python $PYTHON_VER..."
    PY_TGZ="Python-$PYTHON_VER.tgz"
    wget -c "https://www.python.org/ftp/python/$PYTHON_VER/$PY_TGZ"
    tar -xzf "$PY_TGZ"
    cd "Python-$PYTHON_VER"
    ./configure --enable-optimizations
    echo "⚙️ 正在多核加速编译..."
    make -j$(nproc)
    sudo make altinstall
    cd ..
    rm -rf "Python-$PYTHON_VER" && rm -f "$PY_TGZ"
fi

# 3. 创建 venv 虚拟环境
echo "📦 3/7: 正在创建 3.10 虚拟环境..."
$PYTHON_PATH -m venv $ENV_NAME

# 4. 激活环境并初始化
source $ENV_NAME/bin/activate
echo "🆙 4/7: 极速模式升级 pip..."
smart_pip_install "--upgrade pip"

# 5. 直接在线安装 Torch 2.3.0(CU118)
echo "💿 5/7: 正在从官方源安装 Torch 2.3.0 和 Torchvision 0.18.0 (CU118)..."
# 使用官方源直接拉取，避免国内镜像源找不到 +cu118 后缀的包
if pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118; then
    echo "✅ Torch 安装成功。"
else
    echo "❌ 错误: Torch 2.3.0 安装失败，请检查网络。"
    exit 1
fi

# 6. 安装 requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📋 6/7: 正在清理冗余依赖并安装业务清单..."
    # 剔除 Python 3.7+ 已内置的冲突包
    sed -i '/^dataclasses==/d' requirements.txt
    
    # 从 txt 中剔除 torch 相关依赖，防止在下面跑国内镜像源时因找不到 +cu118 再次触发崩溃
    sed -i '/^torch==/d' requirements.txt
    sed -i '/^torchvision==/d' requirements.txt
    sed -i '/^torchaudio==/d' requirements.txt
    
    # 强制使用兼容的 Cython 编译 pycocotools
    echo "🔧 正在自动修复 pycocotools 编译环境..."
    pip uninstall -y Cython numpy
    smart_pip_install "Cython<3.0.0 numpy<2.0.0 wheel"
    smart_pip_install "pycocotools --no-build-isolation --no-cache-dir"
    
    # 从 requirements.txt 中彻底删除 pycocotools
    sed -i '/pycocotools/d' requirements.txt
    
    # 批量安装剩余的安全依赖
    if smart_pip_install "-r requirements.txt"; then
        INSTALL_STATUS=0
    else
        INSTALL_STATUS=1
    fi
else
    INSTALL_STATUS=0
fi

# 7. 清理与效能自检
if [ $INSTALL_STATUS -eq 0 ]; then
    echo "------------------------------------------------"
    echo "🧹 部署成功！执行 CUDA 最终自检..."
    python -c "import torch; print('🚀 CUDA 状态:', torch.cuda.is_available()); print('📦 Torch 版本:', torch.__version__)"
    echo "------------------------------------------------"
else
    echo "⚠️ 依赖安装中途出错。"
    exit 1
fi

exec bash --rcfile <(echo "source ~/.bashrc; source $TARGET_DIR/$ENV_NAME/bin/activate; echo '🔥 环境已就绪！'")
