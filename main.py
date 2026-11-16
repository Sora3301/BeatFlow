#!/usr/bin/env python3
"""BeatFlow 主程序入口。

    python3 main.py demo                        # 内置序列，零数据文件
    python3 main.py run --recorded preds.jsonl  # 回放录好的模型输出
    python3 main.py run --recorded preds.jsonl --backend uinput --lead-in 3
"""

from __future__ import annotations

from beatflow_main.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
