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

codex plugin marketplace upgrade

prd_path="$script_dir/docs/mozi/abs/prd.md"
operator_dir="$(dirname "$prd_path")"
spec_path="$operator_dir/spec.md"
rm -rf "$spec_path"
create_spec_result_path="$operator_dir/.codex-create-spec-result.txt"
review_spec_result_path="$operator_dir/.codex-review-spec-result.yaml"

run_review() {
  codex exec \
    --model gpt-5.5 \
    -c model_reasoning_effort=xhigh \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --output-last-message "$review_spec_result_path" \
    - <<PROMPT
\$mozi:review-spec ${spec_path}
PROMPT
}

review_score() {
  python - "$review_spec_result_path" <<'PY'
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

revise_spec_from_review() {
  codex exec \
    --model gpt-5.5 \
    -c model_reasoning_effort=xhigh \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --output-last-message "$create_spec_result_path" \
    - <<PROMPT
\$mozi:create-spec
请根据 review 文件修改已有 SPEC。

PRD path: ${prd_path}
SPEC path: ${spec_path}
Review file: ${review_spec_result_path}

要求：
- 使用 create-spec Revision Mode。
- 读取 review YAML 中的 review_result.dimensions、critical_issues、recommended_actions、summary 和 grade。
- 以 PRD 作为唯一需求来源，SPEC 只按 review 指出的问题补强。
- 只修改同一个 SPEC 文件，不创建新 SPEC。
- 保持 SPEC 阶段边界，不加入 DESIGN/IMPLEMENT 细节。
PROMPT
}

if [[ ! -r "$prd_path" ]]; then
  echo "Expected PRD not found or unreadable: $prd_path" >&2
  exit 1
fi

codex exec \
  --model gpt-5.5 \
  -c model_reasoning_effort=xhigh \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  --output-last-message "$create_spec_result_path" \
  - <<PROMPT
\$mozi:create-spec
请基于现有 PRD 生成 SPEC。

PRD path: ${prd_path}

要求：
- 使用 create-spec Create Mode。
- 读取 PRD 作为唯一需求来源。
- 生成或覆盖 sibling SPEC: ${spec_path}
- 保持 SPEC 阶段边界，不加入 DESIGN/IMPLEMENT 细节。
PROMPT

if [[ ! -r "$spec_path" ]]; then
  echo "Expected SPEC not found or unreadable: $spec_path" >&2
  exit 1
fi

SPEC_REVIEW_PASSED=false
rerun_count=0

echo "MAX_RERUN_TIMES=${MAX_RERUN_TIMES}"

while true; do
  run_review

  if review_passed; then
    SPEC_REVIEW_PASSED=true
    echo "SPEC_REVIEW_PASSED=${SPEC_REVIEW_PASSED}"
    exit 0
  fi

  current_score="$(review_score)"
  if [[ "$rerun_count" -ge "$MAX_RERUN_TIMES" ]]; then
    echo "SPEC_REVIEW_PASSED=${SPEC_REVIEW_PASSED}"
    echo "Final SPEC review score: ${current_score}"
    echo "Review result: ${review_spec_result_path}"
    exit 1
  fi

  rerun_count=$((rerun_count + 1))
  echo "SPEC review score ${current_score}; revision ${rerun_count}/${MAX_RERUN_TIMES}"
  revise_spec_from_review

  if [[ ! -r "$spec_path" ]]; then
    echo "Expected SPEC not found or unreadable after revision: $spec_path" >&2
    exit 1
  fi
done
