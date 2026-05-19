#!/usr/bin/env bash
clear
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

venv_dir="$script_dir/.venv"
if [[ ! -r "$venv_dir/bin/activate" ]]; then
  echo "Expected Python virtual environment not found: $venv_dir" >&2
  echo "Run: python3 -m venv .venv && .venv/bin/python -m pip install pyyaml" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$venv_dir/bin/activate"

rm -rf docs/
codex plugin marketplace upgrade

prd_path="$script_dir/docs/mozi/abs/prd.md"
operator_dir="$(dirname "$prd_path")"
create_prd_result_path="$operator_dir/.codex-create-prd-result.txt"
review_prd_result_path="$operator_dir/.codex-review-prd-result.yaml"

codex exec \
  --model gpt-5.4-mini \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  --output-last-message "$create_prd_result_path" \
  - <<'PROMPT'
$mozi:create-prd 开发一个 Abs 算子：输入一个 float32 类型 Tensor x，输出一个与输入 shape 相同、dtype 相同的 Tensor y，满足 y[i] = |x[i]|；算子需要支持任意维度 Tensor，并正确处理正数、负数、+0.0、-0.0、NaN、+Inf、-Inf、空 Tensor、标量 Tensor 等边界场景。面向NPU ARCH dav-2201开发算子。
PROMPT

if [[ ! -r "$prd_path" ]]; then
  echo "Expected PRD not found or unreadable: $prd_path" >&2
  exit 1
fi

codex exec \
  --model gpt-5.5 \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  --output-last-message "$review_prd_result_path" \
  - <<PROMPT
\$mozi:review-prd ${prd_path}
PROMPT

review_score="$(
  python - "$review_prd_result_path" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    review = yaml.safe_load(f)

print(review["review_result"]["total_score"])
PY
)"

PRD_REVIEW_PASSED=false
if [[ "$review_score" -gt 95 ]]; then
  PRD_REVIEW_PASSED=true
fi

echo "PRD_REVIEW_PASSED=${PRD_REVIEW_PASSED}"
