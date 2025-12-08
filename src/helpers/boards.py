import json
from typing import List
from urllib.request import urlopen

# Fallback board list (used if API unavailable)
DEFAULT_BOARDS: List[str] = [
    # SFW
    "a",
    "c",
    "w",
    "m",
    "cgl",
    "cm",
    "f",
    "n",
    "jp",
    "vt",
    "vp",
    "v",
    "vg",
    "vr",
    "vm",
    "vmg",
    "vst",
    "co",
    "g",
    "tv",
    "k",
    "o",
    "an",
    "tg",
    "sp",
    "xs",
    "sci",
    "his",
    "int",
    "out",
    "toy",
    "i",
    "po",
    "p",
    "ck",
    "ic",
    "wg",
    "lit",
    "mu",
    "fa",
    "3",
    "gd",
    "diy",
    "wsg",
    "biz",
    "trv",
    "fit",
    "s4s",
    "adv",
    "news",
    "qa",
    "qst",
    # 18+
    "b",
    "r9k",
    "pol",
    "soc",
    "s",
    "hc",
    "hm",
    "h",
    "e",
    "u",
    "d",
    "y",
    "t",
    "hr",
    "gif",
    "aco",
    "trash",
    "mlp",
    "bant",
    "x",
]


def load_4chan_boards() -> List[str]:
    """Load all 4chan board codes from public API with a safe fallback."""
    url = "https://a.4cdn.org/boards.json"
    try:
        with urlopen(url, timeout=6) as resp:
            data = json.load(resp)
            names = sorted([b.get("board") for b in data.get("boards", []) if b.get("board")])
            if names:
                return names
    except Exception:
        pass
    return DEFAULT_BOARDS


__all__ = ["DEFAULT_BOARDS", "load_4chan_boards"]
