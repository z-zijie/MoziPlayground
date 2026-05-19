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

MAX_RERUN_TIMES="${MAX_RERUN_TIMES:-3}"
if ! [[ "$MAX_RERUN_TIMES" =~ ^[0-9]+$ ]]; then
  echo "MAX_RERUN_TIMES must be a non-negative integer: $MAX_RERUN_TIMES" >&2
  exit 1
fi

rm -rf docs/
codex plugin marketplace upgrade

prd_path="$script_dir/docs/mozi/abs/prd.md"
operator_dir="$(dirname "$prd_path")"
create_prd_result_path="$operator_dir/.codex-create-prd-result.txt"
review_prd_result_path="$operator_dir/.codex-review-prd-result.yaml"

run_review() {
  codex exec \
    --model gpt-5.5 \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --output-last-message "$review_prd_result_path" \
    - <<PROMPT
\$mozi:review-prd ${prd_path}
PROMPT
}

review_score() {
  python - "$review_prd_result_path" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    review = yaml.safe_load(f)

print(review["review_result"]["total_score"])
PY
}

review_passed() {
  local score
  score="$(review_score)"
  [[ "$score" -gt 90 ]]
}

revise_prd_from_review() {
  codex exec \
    --model gpt-5.5 \
    -c model_reasoning_effort=high \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --output-last-message "$create_prd_result_path" \
    - <<PROMPT
\$mozi:create-prd
请根据 review 文件修改已有 PRD。

PRD path: ${prd_path}
Review file: ${review_prd_result_path}

要求：
- 使用 create-prd Revision Mode。
- 读取 review YAML 中的 blocking_issues、key_issues、improvement_suggestions、score_breakdown、spec_entry_decision 和 review_notes。
- 只修改同一个 PRD 文件，不创建新 PRD。
- 保持 PRD 阶段边界，不加入 SPEC/DESIGN/IMPLEMENT 细节。
PROMPT
}

codex exec \
  --model gpt-5.5 \
  -c model_reasoning_effort=high \
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

PRD_REVIEW_PASSED=false
rerun_count=0

echo "MAX_RERUN_TIMES=${MAX_RERUN_TIMES}"

while true; do
  run_review

  if review_passed; then
    PRD_REVIEW_PASSED=true
    echo "PRD_REVIEW_PASSED=${PRD_REVIEW_PASSED}"
    exit 0
  fi

  current_score="$(review_score)"
  if [[ "$rerun_count" -ge "$MAX_RERUN_TIMES" ]]; then
    echo "PRD_REVIEW_PASSED=${PRD_REVIEW_PASSED}"
    echo "Final PRD review score: ${current_score}"
    echo "Review result: ${review_prd_result_path}"
    exit 1
  fi

  rerun_count=$((rerun_count + 1))
  echo "PRD review score ${current_score}; revision ${rerun_count}/${MAX_RERUN_TIMES}"
  revise_prd_from_review

  if [[ ! -r "$prd_path" ]]; then
    echo "Expected PRD not found or unreadable after revision: $prd_path" >&2
    exit 1
  fi
done
