#!/bin/bash

# --- Strict configuration ---
ENV_NAME="metafusion_venv"
PYTHON_VER="3.10"
PYTHON_PATH="/usr/local/bin/python3.10"
TARGET_DIR=$(pwd)

# Clear proxy variables to avoid accidental mirror routing.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "------------------------------------------------"
echo "ViR_MFS environment setup (Torch 2.3.0 + CU118 + Python 3.10)"
echo "------------------------------------------------"

# --- Smart pip installer. Prefer PyPI first, then fall back to mirrors. ---
smart_pip_install() {
    local args="$1"
    local mirrors=("https://pypi.org/simple" "https://mirrors.aliyun.com/pypi/simple/" "https://pypi.tuna.tsinghua.edu.cn/simple")
    
    for mirror in "${mirrors[@]}"; do
        echo "Trying package index: $mirror ..."
        if pip install $args -i "$mirror" --timeout 30 --retries 1; then
            return 0
        fi
    done
    return 1
}

# --- Python version detection and optional source build. ---
check_python_exists() {
    if [ -f "$PYTHON_PATH" ] || command -v python3.10 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

if check_python_exists; then
    [ ! -f "$PYTHON_PATH" ] && PYTHON_PATH=$(command -v python3.10)
    echo "Python 3.10 already exists: $PYTHON_PATH"
else
    echo "1/7: Installing Python build dependencies..."
    sudo apt update
    sudo apt install -y build-essential zlib1g-dev libncurses-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev
    echo "2/7: Building Python $PYTHON_VER..."
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

# 3. Create the virtual environment.
echo "3/7: Creating Python 3.10 virtual environment..."
$PYTHON_PATH -m venv $ENV_NAME
source $ENV_NAME/bin/activate

# 4. Upgrade core build tools to avoid wheel/packaging conflicts.
echo "4/7: Upgrading base build tools..."
smart_pip_install "--upgrade pip setuptools wheel packaging"

# 5. Install Torch 2.3.0 with CUDA 11.8 wheels.
echo "5/7: Installing Torch 2.3.0 and Torchvision 0.18.0 (CU118)..."
if pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118; then
    echo "Torch installation succeeded."
else
    echo "Torch installation failed. Check your network or CUDA wheel index."
    exit 1
fi

# 6. Install project dependencies.
if [ -f "requirements.txt" ]; then
    echo "6/7: Installing project dependencies..."
    
    # Build pycocotools with compatible build-time dependencies first.
    echo "Preparing pycocotools build dependencies..."
    smart_pip_install "Cython<3.0.0 numpy<2.0.0"
    smart_pip_install "pycocotools --no-build-isolation --no-cache-dir"
    
    # Install the remaining dependencies with PyPI-first fallback behavior.
    echo "Installing requirements.txt..."
    if smart_pip_install "-r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118"; then
        INSTALL_STATUS=0
    else
        INSTALL_STATUS=1
    fi
else
    INSTALL_STATUS=0
fi

# 7. Final validation.
if [ $INSTALL_STATUS -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Setup completed. Running final validation..."
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Torch version:', torch.__version__)"
    echo "------------------------------------------------"
else
    echo "Dependency installation failed."
    exit 1
fi

exec bash --rcfile <(echo "source ~/.bashrc; source $TARGET_DIR/$ENV_NAME/bin/activate; echo 'Environment is ready.'")
