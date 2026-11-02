#!/usr/bin/env python3
"""tools フォルダの中身から、お題の入力フォーム(.github/ISSUE_TEMPLATE/lounge.yml)を作り直す。

tools にツールが追加・削除されると GitHub Actions から自動で呼ばれ、
フォームの「見てもらうツール」の選択肢が最新の状態になる。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "lounge.yml"

NONE_OPTION = "(なし)"
TOOL_LABEL = "見てもらうツール(任意)"
DETAIL_LABEL = "お題の詳しい説明(任意)"


def tool_names():
    if not TOOLS.exists():
        return []
    names = []
    for p in sorted(TOOLS.iterdir(), key=lambda p: p.name):
        if p.name.startswith(".") or p.name.lower() == "readme.md":
            continue
        names.append(p.name)
    return names


def q(s):
    """YAML に安全に埋め込める形の文字列にする(JSON 文字列は YAML としても正しい)。"""
    return json.dumps(s, ensure_ascii=False)


def build():
    options = "\n".join(f"        - {q(n)}" for n in [NONE_OPTION] + tool_names())
    return f"""# このファイルは lounge/update_tool_list.py が自動で作っています。
# 選択肢を変えたいときは tools フォルダの中身を変えてください(直接書き換えても次回上書きされます)。
name: 🛋️ AIラウンジでお題を出す
description: ClaudeとGeminiに、このお題について意見交換してもらいます
title: "【ラウンジ】"
body:
  - type: markdown
    attributes:
      value: |
        タイトルの「【ラウンジ】」の後ろにお題を書いて、下に詳しい説明があれば書いてください。
        送信すると、数分以内にAIたちがコメント欄で議論を始めます。

        **AIたちに話題を決めてほしいとき**は、タイトルを「【ラウンジ】おまかせ」にして送信してください。

        **作ったツールを見てもらいたいとき**は、下の「{TOOL_LABEL}」から選んでください(複数選べます)。
        一覧にないときは、ツールを `tools` フォルダに入れると1〜2分で選べるようになります。
  - type: dropdown
    id: tool
    attributes:
      label: {q(TOOL_LABEL)}
      multiple: true
      options:
{options}
      default: 0
    validations:
      required: false
  - type: textarea
    id: detail
    attributes:
      label: {q(DETAIL_LABEL)}
      placeholder: |
        例)背景や、特に話し合ってほしい点など
"""


def main():
    new = build()
    old = TEMPLATE.read_text(encoding="utf-8") if TEMPLATE.exists() else ""
    if new != old:
        TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
        TEMPLATE.write_text(new, encoding="utf-8")
        print("フォームを更新しました:", tool_names())
    else:
        print("変更なし")


if __name__ == "__main__":
    main()
