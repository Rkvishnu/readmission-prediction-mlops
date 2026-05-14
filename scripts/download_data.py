from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from readmission.utils.config import load_yaml
from readmission.utils.paths import CONFIG_DIR


def download_and_extract(source_url: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "diabetes_130_us_hospitals.zip"
    urlretrieve(source_url, archive_path)

    with ZipFile(archive_path) as archive:
        archive.extractall(output_dir)

    return sorted(path for path in output_dir.iterdir() if path.is_file())


def main() -> None:
    data_config = load_yaml(CONFIG_DIR / "data_config.yaml")
    output_files = download_and_extract(data_config["source_url"], Path("data/raw"))
    print("Downloaded dataset files:")
    for path in output_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
