"""把报告 Markdown 转成可直接发人的单文件 HTML（自带样式，可直接打印成 PDF）。

为什么不用 pandoc/Word：本机只有 `markdown` 库，没有 python-docx / pypandoc。
单文件 HTML 的好处是双击就能看、发微信/邮件不会被防火墙拦、浏览器里"打印→另存为 PDF"
即可得到排版正确的 PDF。

用法：
    python backend/tools/md2html.py docs/实验报告_g1_20261124.md
    python backend/tools/md2html.py in.md -o out.html --title "自定义标题"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown

CSS = """
:root {
  --fg: #1f2328; --dim: #57606a; --line: #d0d7de; --bg: #ffffff;
  --accent: #0b6bcb; --soft: #f6f8fa; --warn: #b45309; --err: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 72px; background: var(--bg); color: var(--fg);
  font: 15px/1.75 -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
}
main { max-width: 900px; margin: 0 auto; }
h1 { font-size: 26px; line-height: 1.3; margin: 0 0 4px; }
h2 {
  font-size: 20px; margin: 34px 0 12px; padding-bottom: 6px;
  border-bottom: 2px solid var(--line);
}
h3 { font-size: 16px; margin: 24px 0 8px; color: var(--accent); }
p, li { margin: 8px 0; }
ul, ol { padding-left: 22px; }
strong { font-weight: 650; }
code {
  font: 0.9em/1.4 "Cascadia Code", Consolas, "SF Mono", monospace;
  background: var(--soft); padding: 1px 4px; border-radius: 4px;
}
pre {
  background: var(--soft); padding: 12px 14px; border-radius: 8px;
  overflow-x: auto; border: 1px solid var(--line);
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 12px 0 18px;
  font-size: 13.5px; display: block; overflow-x: auto;
}
th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: left; }
th { background: var(--soft); font-weight: 650; white-space: nowrap; }
tbody tr:nth-child(even) { background: #fcfcfd; }
blockquote {
  margin: 12px 0; padding: 8px 14px; border-left: 4px solid var(--accent);
  background: var(--soft); color: var(--dim);
}
hr { border: none; border-top: 1px solid var(--line); margin: 28px 0; }
.meta { color: var(--dim); font-size: 13px; margin-bottom: 18px; }

@media print {
  body { padding: 0; font-size: 11pt; }
  main { max-width: none; }
  h2 { page-break-after: avoid; }
  table, pre, blockquote { page-break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
}
"""

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body><main>
{body}
</main></body>
</html>
"""


def convert(md_text: str, title: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "sane_lists", "fenced_code", "attr_list"],
        output_format="html5",
    )
    return HTML.format(title=title, css=CSS, body=body)


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown -> 单文件 HTML 报告")
    ap.add_argument("source")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_file():
        print(f"找不到文件: {src}")
        return 2
    text = src.read_text(encoding="utf-8")
    title = args.title
    if not title:
        m = re.search(r"^#\s+(.+)$", text, re.M)
        title = m.group(1).strip() if m else src.stem

    out = Path(args.out) if args.out else src.with_suffix(".html")
    out.write_text(convert(text, title), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"  {src.name} -> {out.name}  ({kb:.1f} KB)")
    print(f"  标题: {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
