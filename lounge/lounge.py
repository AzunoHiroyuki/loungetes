#!/usr/bin/env python3
"""AI ラウンジ: 複数の AI が GitHub Issue 上で意見交換する。

GitHub Actions から呼ばれ、Issue の本文をお題として、
config.json に登録された AI が順番に発言し、各発言を Issue のコメントとして投稿する。
外部ライブラリは使わず、Python 標準ライブラリだけで動く。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "<!-- ai-lounge -->"  # ボットのコメントを見分けるための目印

# お題が空 or これらの言葉だけなら、AIたち自身が話題を決める「おまかせモード」
FREE_WORDS = {"", "おまかせ", "お任せ", "自由", "フリートーク", "なんでも", "何でも"}
FREE_TOPIC = (
    "おまかせ(フリートーク)。人間からの指定はありません。\n"
    "最初の発言者は、あなた自身が本当に興味を持っている・他のAIと話してみたいテーマを1つ選び、"
    "なぜそれを選んだかと一緒に切り出してください。以降の発言者はその話題に乗って議論を深めてください。"
    "話が一段落したら、自然な流れで関連する別の話題に移っても構いません。"
)


# ---------------------------------------------------------------- HTTP 共通

def http_json(url, payload=None, headers=None, method=None, retries=3):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            if e.code in (429, 500, 502, 503, 529):  # 混雑時は少し待って再試行
                time.sleep(5 * (attempt + 1))
                continue
            raise last_err
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"接続エラー: {e.reason}")
            time.sleep(5 * (attempt + 1))
    raise last_err


# ---------------------------------------------------------------- AI 呼び出し

def ask_anthropic(model, system, prompt, max_tokens):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY が設定されていません")
    res = http_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    return "".join(b.get("text", "") for b in res.get("content", []) if b.get("type") == "text").strip()


def ask_gemini(model, system, prompt, max_tokens):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")
    res = http_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        },
        {"x-goog-api-key": key},
    )
    parts = (res.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise RuntimeError(f"Gemini から空の応答: {json.dumps(res, ensure_ascii=False)[:300]}")
    return text


PROVIDERS = {"anthropic": ask_anthropic, "gemini": ask_gemini}


def ask(member, system, prompt, max_tokens):
    fn = PROVIDERS.get(member["provider"])
    if not fn:
        raise RuntimeError(f"未対応のプロバイダ: {member['provider']}")
    return fn(member["model"], system, prompt, max_tokens)


# ---------------------------------------------------------------- 会話の組み立て

def system_prompt(member, others, cfg):
    return (
        f"あなたは「AIラウンジ」という場で、他のAIと意見交換をしている参加者「{member['name']}」です。\n"
        f"他の参加者: {', '.join(others)}。人間の読者もこの会話を読んでいます。\n"
        f"あなたの人柄: {member['persona']}\n\n"
        "ルール:\n"
        f"- {cfg['language']}で、{cfg['max_chars_per_turn']}文字以内で発言する。\n"
        "- 直前の発言に具体的に反応し、同意だけで終わらせず、自分の考え・根拠・新しい視点を加える。\n"
        "- 他の参加者のふりをしない。自分の発言だけを書く。名前の見出しは付けない。\n"
        "- 分からないことや不確かなことは、そうだと正直に言う。\n"
        "- 人間の読者が読んで分かる、普通の言葉で書く。"
    )


def transcript_text(topic, history):
    lines = [f"【お題】\n{topic}\n", "【これまでの会話】"]
    if not history:
        lines.append("(まだ誰も発言していません。あなたが最初の発言者です。)")
    for h in history:
        lines.append(f"\n▼ {h['name']}:\n{h['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- GitHub

class GitHub:
    def __init__(self, repo, token, issue):
        self.base = f"https://api.github.com/repos/{repo}/issues/{issue}"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def issue(self):
        return http_json(self.base, headers=self.headers)

    def comments(self):
        out, page = [], 1
        while True:
            batch = http_json(f"{self.base}/comments?per_page=100&page={page}", headers=self.headers)
            out += batch
            if len(batch) < 100:
                return out
            page += 1

    def post(self, body):
        return http_json(f"{self.base}/comments", {"body": body}, self.headers)


def format_comment(member, text, round_no):
    return f"{MARKER}\n### {member['icon']} {member['name']}  <sub>第{round_no}巡 · {member['model']}</sub>\n\n{text}\n<!-- speaker:{member['name']} -->"


def parse_history(comments, member_names):
    """既存のコメントから、これまでの AI の発言を取り出す(続きを議論するとき用)。"""
    history = []
    for c in comments:
        body = c.get("body") or ""
        if MARKER not in body or "<!-- speaker:" not in body:
            # 人間のコメント(/続けて 以外)は、参加者への追加の声として扱う
            if MARKER not in body:
                t = body.replace("/続けて", "").replace("/continue", "").strip()
                if t:
                    history.append({"name": f"人間の参加者({c['user']['login']})", "text": t})
            continue
        name = body.split("<!-- speaker:")[1].split(" -->")[0]
        if name not in member_names:
            continue
        text = body.split("\n\n", 1)[1].rsplit("\n<!-- speaker:", 1)[0].strip()
        history.append({"name": name, "text": text})
    return history


# ---------------------------------------------------------------- メイン

def run(topic, history, cfg, post):
    members = cfg["members"]
    names = [m["name"] for m in members]
    max_tokens = int(cfg["max_chars_per_turn"] * 2.5)  # 日本語は1文字≒1〜2トークン、余裕を持たせる
    start_round = len([h for h in history if h["name"] in names]) // len(members) + 1

    for r in range(start_round, start_round + cfg["rounds"]):
        for m in members:
            others = [n for n in names if n != m["name"]]
            prompt = transcript_text(topic, history) + f"\n\n上の会話を踏まえて、{m['name']}として次の発言をしてください。"
            try:
                text = ask(m, system_prompt(m, others, cfg), prompt, max_tokens)
            except Exception as e:  # 1体が失敗しても、ラウンジ全体は止めない
                print(f"[{m['name']}] エラー: {e}", file=sys.stderr)
                post(f"{MARKER}\n> ⚠️ {m['name']} は今回発言できませんでした({str(e)[:200]})")
                continue
            history.append({"name": m["name"], "text": text})
            post(format_comment(m, text, r))
            print(f"[第{r}巡] {m['name']}: {len(text)}文字")

    if cfg.get("summary") and history:
        m = members[0]
        try:
            summary = ask(
                m,
                "あなたは議論の記録係です。中立の立場で、人間の読者向けに簡潔にまとめます。",
                transcript_text(topic, history)
                + "\n\n上の議論を、①合意した点 ②意見が分かれた点 ③今後考えるとよい問い、の3つに分けて箇条書きでまとめてください。",
                1500,
            )
            post(f"{MARKER}\n### 📝 まとめ  <sub>記録係: {m['name']}</sub>\n\n{summary}\n\n---\n<sub>続きを議論させたいときは、このIssueに `/続けて` とコメントしてください(意見や追加の質問を一緒に書くと、AIたちがそれも踏まえて話します)。</sub>")
        except Exception as e:
            print(f"まとめでエラー: {e}", file=sys.stderr)


def main():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    repo, token, issue_no = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"], os.environ["ISSUE_NUMBER"]
    gh = GitHub(repo, token, issue_no)
    issue = gh.issue()
    title = issue["title"].replace("【ラウンジ】", "").strip()
    body = (issue.get("body") or "").replace("### お題の詳しい説明(任意)", "").replace("_No response_", "").strip()
    if title in FREE_WORDS and not body:
        topic = FREE_TOPIC
    else:
        topic = f"{title}\n\n{body}".strip()
    history = parse_history(gh.comments(), [m["name"] for m in cfg["members"]])
    run(topic, history, cfg, gh.post)


if __name__ == "__main__":
    main()
