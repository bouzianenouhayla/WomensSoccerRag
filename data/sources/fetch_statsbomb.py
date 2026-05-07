from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
TARGET_DIR = Path(__file__).resolve().parent / "statsbomb"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

RESOURCES: Dict[str, str] = {
    "competitions": f"{BASE_URL}/competitions.json",
    "matches_fifa_wwc_2019": f"{BASE_URL}/matches/72/30.json",  # 2019 WWC
    "lineups_fifa_wwc_final": f"{BASE_URL}/lineups/22912.json",
}


def download_resource(name: str, url: str) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    with open(TARGET_DIR / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Saved {name} to {TARGET_DIR}")


def main() -> None:
    for name, url in RESOURCES.items():
        download_resource(name, url)


if __name__ == "__main__":
    main()
