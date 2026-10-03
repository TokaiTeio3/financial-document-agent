#!/usr/bin/env bash

# 使用既有 processed_data 复现 B 榜性能。
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_PATH=""
OUTPUT_PATH=""
CONFIG_PATH="${SCRIPT_DIR}/config/config.ultra.yaml"
WORKERS="4"
BOOTSTRAP_PYTHON=""
SKIP_INSTALL="false"

show_help() {
  cat <<'EOF'
用法：
  ./reproduce.sh --input <性能复现包根目录> --output <空输出目录> [选项]

必需参数：
  --input PATH          包含 processed_data 和 data/raw_dataset/questions 的目录
  --output PATH         不存在或为空的性能复现输出目录

可选参数：
  --config PATH         配置文件，默认 config/config.ultra.yaml
  --workers N           API 并发数，默认 4
  --python PATH         用于创建虚拟环境的 Python 解释器
  --skip-install        复用现有 .venv，跳过依赖安装
  -h, --help            显示帮助

本脚本不执行文档预处理。运行前请设置 DASHSCOPE_API_KEY。
EOF
}

fail() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --input)
      (($# >= 2)) || fail "--input 缺少参数"
      INPUT_PATH="$2"
      shift 2
      ;;
    --output)
      (($# >= 2)) || fail "--output 缺少参数"
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --config)
      (($# >= 2)) || fail "--config 缺少参数"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --workers)
      (($# >= 2)) || fail "--workers 缺少参数"
      WORKERS="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || fail "--python 缺少参数"
      BOOTSTRAP_PYTHON="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL="true"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      fail "未知参数：$1"
      ;;
  esac
done

[[ -n "${INPUT_PATH}" ]] || fail "必须提供 --input"
[[ -n "${OUTPUT_PATH}" ]] || fail "必须提供 --output"
[[ -d "${INPUT_PATH}" ]] || fail "性能复现包目录不存在：${INPUT_PATH}"
[[ -d "${INPUT_PATH}/processed_data" ]] || fail "缺少 ${INPUT_PATH}/processed_data"
[[ -d "${INPUT_PATH}/data/raw_dataset/questions" ]] || fail "缺少 ${INPUT_PATH}/data/raw_dataset/questions"
[[ -f "${CONFIG_PATH}" ]] || fail "配置文件不存在：${CONFIG_PATH}"
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || fail "--workers 必须是正整数"
[[ -n "${DASHSCOPE_API_KEY:-}" ]] || fail "请先设置 DASHSCOPE_API_KEY"

cd "${SCRIPT_DIR}"

if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v python)"
  else
    fail "未找到 Python 3.11 或更高版本"
  fi
fi

if [[ ! -d "${SCRIPT_DIR}/.venv" ]]; then
  printf '正在创建虚拟环境：%s\n' "${SCRIPT_DIR}/.venv"
  "${BOOTSTRAP_PYTHON}" -m venv "${SCRIPT_DIR}/.venv"
fi

if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  RUNTIME_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
elif [[ -x "${SCRIPT_DIR}/.venv/Scripts/python.exe" ]]; then
  RUNTIME_PYTHON="${SCRIPT_DIR}/.venv/Scripts/python.exe"
else
  fail "虚拟环境中的 Python 不可用，请删除 .venv 后重试"
fi

"${RUNTIME_PYTHON}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
  || fail "需要 Python 3.11 或更高版本"

if [[ "${SKIP_INSTALL}" != "true" ]]; then
  printf '正在安装或校验依赖……\n'
  "${RUNTIME_PYTHON}" -m pip install --disable-pip-version-check -r "${SCRIPT_DIR}/requirements.txt"
fi

printf '开始 B 榜性能复现，输出目录：%s\n' "${OUTPUT_PATH}"
"${RUNTIME_PYTHON}" "${SCRIPT_DIR}/run_reproduce.py" \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --config "${CONFIG_PATH}" \
  --workers "${WORKERS}"
