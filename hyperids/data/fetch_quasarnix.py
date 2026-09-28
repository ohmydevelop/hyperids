"""Fetch the QuasarNix malicious-command corpus (public dataset).

TODO(data-source): 当前未提供 QuasarNix 的 HF/GitHub 地址。
  补全后实现 `_fetch()`，输出与 dataset/corpus/raw/public/quasarnix.jsonl 对齐：
     每行 {"text": <command>, "source": "quasarnix"}
  之后由 `hyperids.jev_labels` 打 31 维 Jev 标签。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "corpus" / "raw" / "public" / "quasarnix.jsonl"

# TODO: 填入 QuasarNix 数据源的下载地址（HF dataset / GitHub raw / ...）
QUASARNIX_SOURCE_URL = "TODO"


def _fetch() -> list[str]:
    """返回 QuasarNix 恶意命令文本列表（待实现）。"""
    raise NotImplementedError(
        "QuasarNix 数据源地址未配置；请填入 QUASARNIX_SOURCE_URL 后实现 _fetch()"
    )


def main():
    cmds = _fetch()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in cmds:
            f.write('{"text": ' + __import__("json").dumps(c.strip(), ensure_ascii=False)
                    + ', "source": "quasarnix"}\n')
    print(f"wrote {len(cmds)} commands -> {OUT}")


if __name__ == "__main__":
    main()
