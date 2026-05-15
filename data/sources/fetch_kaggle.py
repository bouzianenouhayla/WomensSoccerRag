import zipfile
from pathlib import Path

from kaggle import KaggleApi

DATASET = "josephvm/womens-world-cup-1991-2023"
TARGET_DIR = Path(__file__).resolve().parent / "kaggle"
TARGET_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    api = KaggleApi()
    api.authenticate()
    zip_path = TARGET_DIR / "womens_world_cup.zip"
    api.dataset_download_files(
        DATASET, path=str(TARGET_DIR), filename=str(zip_path.name), quiet=False
    )
    return zip_path


def extract(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(TARGET_DIR)
    zip_path.unlink(missing_ok=True)


def main() -> None:
    zip_path = download_dataset()
    extract(zip_path)
    print(f"Dataset downloaded to {TARGET_DIR}")


if __name__ == "__main__":
    main()
