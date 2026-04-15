#!/bin/bash
#
# SciBERT NER Service 启动脚本
# 兼容 macOS / Linux / Windows (Git Bash/WSL)
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检测操作系统
detect_os() {
    case "$(uname -s)" in
        Darwin*)  OS="macos" ;;
        Linux*)   OS="linux" ;;
        MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
        *)        OS="unknown" ;;
    esac
    print_info "Detected OS: $OS"
}

# 检测包管理器
detect_package_manager() {
    if [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            PKG_MANAGER="brew"
        else
            PKG_MANAGER="none"
        fi
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            PKG_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PKG_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            PKG_MANAGER="dnf"
        else
            PKG_MANAGER="none"
        fi
    else
        PKG_MANAGER="none"
    fi
}

# 安装 Homebrew (macOS)
install_homebrew() {
    if [[ "$OS" == "macos" ]] && ! command -v brew &> /dev/null; then
        print_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        PKG_MANAGER="brew"
    fi
}

# 检查并安装 Python
check_python() {
    print_info "Checking Python installation..."
    
    # 尝试不同的 Python 命令
    PYTHON_CMD=""
    for cmd in python3 python; do
        if command -v $cmd &> /dev/null; then
            version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            major=$(echo $version | cut -d. -f1)
            minor=$(echo $version | cut -d. -f2)
            if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 9 ]]; then
                PYTHON_CMD=$cmd
                print_success "Found Python $version ($cmd)"
                break
            fi
        fi
    done
    
    if [[ -z "$PYTHON_CMD" ]]; then
        print_warning "Python 3.9+ not found, attempting to install..."
        install_python
    fi
}

# 安装 Python
install_python() {
    case "$PKG_MANAGER" in
        brew)
            print_info "Installing Python via Homebrew..."
            brew install python@3.11
            PYTHON_CMD="python3"
            ;;
        apt)
            print_info "Installing Python via apt..."
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3-pip
            PYTHON_CMD="python3.11"
            ;;
        yum|dnf)
            print_info "Installing Python via $PKG_MANAGER..."
            sudo $PKG_MANAGER install -y python3.11 python3.11-pip
            PYTHON_CMD="python3.11"
            ;;
        *)
            print_error "Cannot auto-install Python. Please install Python 3.9+ manually."
            print_info "Download from: https://www.python.org/downloads/"
            exit 1
            ;;
    esac
    print_success "Python installed successfully"
}

# 创建虚拟环境
setup_venv() {
    VENV_DIR="backend/venv"
    
    if [[ ! -d "$VENV_DIR" ]]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # 激活虚拟环境
    if [[ "$OS" == "windows" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
    print_success "Virtual environment activated"
}

# 安装依赖
install_dependencies() {
    print_info "Installing dependencies..."
    
    # 升级 pip
    pip install --upgrade pip -q
    
    # 安装依赖
    pip install -r backend/requirements.txt -q
    
    print_success "Dependencies installed"
}

# 启动服务
start_service() {
    print_info "Starting SciBERT NER Service..."
    echo ""
    echo "=========================================="
    echo "  SciBERT NER Service"
    echo "  API: http://localhost:8000"
    echo "  Docs: http://localhost:8000/docs"
    echo "=========================================="
    echo ""
    
    cd backend
    uvicorn app.main:app --host 0.0.0.0 --port 8000
}

# 主流程
main() {
    echo ""
    echo "=========================================="
    echo "  SciBERT NER Service Setup"
    echo "=========================================="
    echo ""
    
    detect_os
    detect_package_manager
    
    # macOS 自动安装 Homebrew
    if [[ "$OS" == "macos" ]] && [[ "$PKG_MANAGER" == "none" ]]; then
        install_homebrew
    fi
    
    check_python
    setup_venv
    install_dependencies
    start_service
}

main "$@"
