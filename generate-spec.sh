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

codex plugin marketplace upgrade

prd_path="$script_dir/docs/mozi/abs/prd.md"
operator_dir="$(dirname "$prd_path")"
spec_path="$operator_dir/spec.md"
rm -rf spec_path
create_spec_result_path="$operator_dir/.codex-create-spec-result.txt"

if [[ ! -r "$prd_path" ]]; then
  echo "Expected PRD not found or unreadable: $prd_path" >&2
  exit 1
fi

codex exec \
  --model gpt-5.4-mini \
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
