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

class QuotaExceeded(RuntimeError):
    """利用枠(無料枠や残高)を使い切った。待っても今回の実行中は回復しないので再試行しない。"""


def describe_quota(detail):
    """Gemini などの 429 応答から、どの枠を超えたかを日本語で説明する。"""
    try:
        err = json.loads(detail).get("error", {})
    except Exception:
        return "利用枠を超えました"
    ids = []
    for d in err.get("details", []):
        for v in d.get("violations", []):
            ids.append(f"{v.get('quotaId', '')}(上限 {v.get('quotaValue', '?')})")
    kinds = " / ".join(ids)
    if "PerDay" in kinds:
        return f"1日あたりの利用枠を使い切りました: {kinds}"
    if "PerMinute" in kinds:
        return f"1分あたりの利用枠を超えました: {kinds}"
    return f"利用枠を超えました: {kinds or err.get('message', '')[:120]}"


def http_json(url, payload=None, headers=None, method=None, retries=4, timeout=150):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 429 and ("quota" in detail.lower() or "credit balance" in detail.lower()):
                # 1日の枠や残高切れは、待っても回復しないのですぐ諦める
                if "PerMinute" in detail and attempt < retries - 1:
                    time.sleep(65)  # 1分あたりの枠なら、1分待てば回復する
                    continue
                raise QuotaExceeded(describe_quota(detail))
            last_err = RuntimeError(f"HTTP {e.code}: {detail[:300]}")
            if e.code in (429, 500, 502, 503, 504, 529):  # 混雑時は待つ時間を延ばしながら再試行
                time.sleep(10 * (attempt + 1))
                continue
            raise last_err
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"接続エラー: {e.reason}")
            time.sleep(10 * (attempt + 1))
        except (TimeoutError, OSError) as e:  # 応答待ちの時間切れも再試行する
            last_err = RuntimeError(f"応答待ちの時間切れ: {e}")
            time.sleep(10 * (attempt + 1))
    raise last_err


# ---------------------------------------------------------------- AI 呼び出し

class Prompt(str):
    """AIに渡す文章。head(お題やプログラム=毎回同じ部分)と残りに分けて持つ。

    Claude には head を「キャッシュ」として渡すので、大きなプログラムを毎回読ませても
    2回目以降は約1/10の料金で済む。普通の文字列としても使える。
    """

    def __new__(cls, head, tail):
        obj = super().__new__(cls, head + tail)
        obj.head, obj.tail = head, tail
        return obj


def to_anthropic_content(prompt):
    if isinstance(prompt, Prompt) and len(prompt.head) > 4000:
        return [
            {"type": "text", "text": prompt.head, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt.tail},
        ]
    return str(prompt)


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
            "messages": [{"role": "user", "content": to_anthropic_content(prompt)}],
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
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
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


# ---------------------------------------------------------------- ツールのレビュー用

REPO_ROOT = HERE.parent
CODE_LIMIT = 250000  # AIに渡すプログラムの合計文字数の上限(料金を抑えるため。config.json の code_limit で変更可)
TEXT_EXT = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm", ".css", ".json", ".md", ".txt",
    ".yml", ".yaml", ".toml", ".ini", ".cfg", ".csv", ".sh", ".bat", ".ps1", ".gs", ".vba",
    ".bas", ".java", ".kt", ".swift", ".go", ".rs", ".rb", ".php", ".c", ".cpp", ".h", ".cs", ".sql", ".xml",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def find_targets(body):
    """説明欄の「対象: tools/〇〇」の行から、見てほしい場所を取り出す。"""
    targets = []
    for line in body.splitlines():
        line = line.strip().lstrip("-・*").strip()
        for head in ("対象:", "対象\uff1a", "target:"):  # 半角・全角どちらのコロンでもOK
            if line.lower().startswith(head):
                targets.append(line[len(head):].strip().strip("`").strip("/"))
    return list(dict.fromkeys(t for t in targets if t))  # 重複を除く


def load_code(targets):
    """対象のファイル・フォルダを読み、AIに渡す文章にまとめる。"""
    files, notes = [], []
    for t in targets:
        p = (REPO_ROOT / t).resolve()
        if REPO_ROOT not in p.parents and p != REPO_ROOT or not p.exists():
            notes.append(f"「{t}」が見つかりませんでした")
            continue
        cands = [p] if p.is_file() else sorted(
            f for f in p.rglob("*") if f.is_file() and not (set(f.relative_to(REPO_ROOT).parts) & SKIP_DIRS)
        )
        for f in cands:
            if f.suffix.lower() in TEXT_EXT:
                if f not in files:
                    files.append(f)
            else:
                notes.append(f"{f.relative_to(REPO_ROOT)}(文字ではないファイルのため省略)")

    parts, used = [], 0
    for f in files:
        rel = f.relative_to(REPO_ROOT)
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="cp932", errors="replace")  # Windowsで作ったファイル向け
        if used + len(text) > CODE_LIMIT:
            notes.append(f"{rel} 以降は長すぎるため省略(上限 {CODE_LIMIT} 文字)")
            break
        parts.append(f"===== ファイル: {rel} =====\n{text}")
        used += len(text)
    return "\n\n".join(parts), notes


def parse_form(body):
    """お題フォームの本文を「説明」と「選ばれたツール」に分ける。

    フォームから作られた本文は「### 見出し\n\n値」の形で並ぶ。
    フォームを使わずに書かれた本文は、まるごと説明として扱う。
    """
    from update_tool_list import TOOL_LABEL, DETAIL_LABEL, NONE_OPTION
    if "### " not in body:
        return body.strip(), []
    sections, current = {}, None
    for line in body.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    get = lambda k: "\n".join(sections.pop(k, [])).replace("_No response_", "").strip()
    tools_raw = get(TOOL_LABEL)
    picked = [t.strip() for t in tools_raw.split(",") if t.strip() and t.strip() != NONE_OPTION]
    detail = get(DETAIL_LABEL)
    rest = "\n\n".join(f"{k}\n" + "\n".join(v).strip() for k, v in sections.items()).strip()
    return "\n\n".join(x for x in (detail, rest) if x), picked


REVIEW_GUIDE = (
    "\n\n【レビューの進め方】\n"
    "上のプログラムは人間が過去に作ったツールです。中身をよく読み、改良案を出し合ってください。\n"
    "- どのファイルのどの部分の話か、具体的に示す\n"
    "- 使いやすさ・壊れにくさ・機能追加などの観点で、理由と一緒に提案する\n"
    "- 相手の案には、賛成・反対・改善をはっきり返す\n"
    "- コードの書き換え例を出すときは、短い部分だけにする\n"
    "- 作った人はプログラムに詳しくない可能性があるので、専門用語は避けるか、かみ砕く"
)


def transcript_text(topic, history, instruction=""):
    head = f"【お題】\n{topic}\n\n"
    lines = ["【これまでの会話】"]
    if not history:
        lines.append("(まだ誰も発言していません。あなたが最初の発言者です。)")
    for h in history:
        lines.append(f"\n▼ {h['name']}:\n{h['text']}")
    return Prompt(head, "\n".join(lines) + (f"\n\n{instruction}" if instruction else ""))


# ---------------------------------------------------------------- GitHub

class GitHub:
    def __init__(self, repo, token, issue):
        self.repo_base = f"https://api.github.com/repos/{repo}"
        self.base = f"{self.repo_base}/issues/{issue}"
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

    def edit(self, comment_id, body):
        return http_json(f"{self.repo_base}/issues/comments/{comment_id}", {"body": body}, self.headers, method="PATCH")

    def react(self, content, comment_id=None):
        """👀 などの絵文字リアクションを付ける(失敗しても議論は止めない)。"""
        url = f"{self.repo_base}/issues/comments/{comment_id}/reactions" if comment_id else f"{self.base}/reactions"
        try:
            http_json(url, {"content": content}, self.headers, retries=1)
        except Exception as e:
            print(f"リアクション失敗: {e}", file=sys.stderr)


def status_text(state, detail, run_url):
    link = f"  <sub>[実行の記録を見る]({run_url})</sub>" if run_url else ""
    return f"{MARKER}\n> {state} {detail}{link}"


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

def run(topic, history, cfg, post, progress=lambda msg: None):
    members = cfg["members"]
    names = [m["name"] for m in members]
    max_tokens = int(cfg["max_chars_per_turn"] * 2.5)  # 日本語は1文字≒1〜2トークン、余裕を持たせる
    start_round = len([h for h in history if h["name"] in names]) // len(members) + 1

    out_of_quota = set()  # 利用枠を使い切ったメンバーは、この回はもう呼ばない
    for r in range(start_round, start_round + cfg["rounds"]):
        active = [m for m in members if m["name"] not in out_of_quota]
        if len(active) < min(2, len(members)):  # 相手がいなくなったら独り言にせず終える
            break
        for m in active:
            others = [n for n in names if n != m["name"]]
            prompt = transcript_text(topic, history, f"上の会話を踏まえて、{m['name']}として次の発言をしてください。")
            try:
                progress(f"第{r}巡:{m['icon']} {m['name']} が考え中…")
                text = ask(m, system_prompt(m, others, cfg), prompt, max_tokens)
            except QuotaExceeded as e:
                out_of_quota.add(m["name"])
                print(f"[{m['name']}] 利用枠切れ: {e}", file=sys.stderr)
                post(f"{MARKER}\n> 🪫 {m['name']} は利用枠を使い切ったため、今回はここで退席します。\n> {e}\n> (Geminiの無料枠は毎日 日本時間17時ごろ〔夏は16時ごろ〕にリセットされます)")
                continue
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
                transcript_text(
                    topic, history,
                    "上の議論を、①合意した点 ②意見が分かれた点 ③今後考えるとよい問い、の3つに分けて箇条書きでまとめてください。"
                    + ("ツールのレビューの場合は、③の代わりに「すぐできる改良」「大きめの改良」に分けて優先順位を付けてください。" if "【見てほしいプログラム】" in topic else ""),
                ),
                1500,
            )
            post(f"{MARKER}\n### 📝 まとめ  <sub>記録係: {m['name']}</sub>\n\n{summary}\n\n---\n<sub>続きを議論させたいときは、このIssueに `/続けて` とコメントしてください(意見や追加の質問を一緒に書くと、AIたちがそれも踏まえて話します)。</sub>")
        except Exception as e:
            print(f"まとめでエラー: {e}", file=sys.stderr)


def main():
    global CODE_LIMIT
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    CODE_LIMIT = int(cfg.get("code_limit", CODE_LIMIT))
    repo, token, issue_no = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"], os.environ["ISSUE_NUMBER"]
    gh = GitHub(repo, token, issue_no)
    issue = gh.issue()
    title = issue["title"].replace("【ラウンジ】", "").strip()
    detail, picked = parse_form(issue.get("body") or "")
    if title in FREE_WORDS and not detail and not picked:
        topic = FREE_TOPIC
    else:
        topic = f"{title}\n\n{detail}".strip()

    # フォームで選んだツール、または説明欄の「対象: tools/〇〇」を読み込んでレビューしてもらう
    review_notes = []
    targets = list(dict.fromkeys([f"tools/{n}" for n in picked] + find_targets(detail)))
    if targets:
        code, review_notes = load_code(targets)
        if code:
            topic += f"\n\n【見てほしいプログラム】\n{code}{REVIEW_GUIDE}"
        if review_notes:
            topic += "\n\n(読み込みメモ: " + " / ".join(review_notes) + ")"
    history = parse_history(gh.comments(), [m["name"] for m in cfg["members"]])

    # 動いていることが分かるように: きっかけのコメント(またはIssue)に 👀 を付け、
    # 進み具合を書き換えていく「状況」コメントを1つ置く
    comment_id = os.environ.get("COMMENT_ID") or None
    run_url = os.environ.get("RUN_URL", "")
    gh.react("eyes", comment_id)
    opening = "ラウンジを開いています…"
    if targets:
        n = topic.count("===== ファイル:")
        opening += f" 📂 プログラムを {n} ファイル読み込みました。"
        if review_notes:
            opening += "(" + " / ".join(review_notes)[:300] + ")"
    status = gh.post(status_text("⏳", opening, run_url))

    def progress(msg):
        try:
            gh.edit(status["id"], status_text("⏳", f"議論中 — {msg}", run_url))
        except Exception as e:
            print(f"状況の更新に失敗: {e}", file=sys.stderr)

    try:
        run(topic, history, cfg, gh.post, progress)
    except Exception as e:
        gh.edit(status["id"], status_text("❌", f"途中で止まりました({str(e)[:150]})", run_url))
        raise
    gh.edit(status["id"], status_text("✅", "今回の議論が終わりました。下に発言とまとめが並んでいます。", run_url))
    gh.react("rocket", comment_id)


if __name__ == "__main__":
    main()
