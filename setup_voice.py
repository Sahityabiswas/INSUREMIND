"""Install the official small English ASR model on the project's drive."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import urllib.request
import zipfile

from src.speech import ROOT, file_sha256

MODEL = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{MODEL}.zip"


def install_model(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"{MODEL}.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=180) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            target = (folder / entry.filename).resolve()
            if not target.is_relative_to(folder) or not entry.filename.startswith(MODEL + "/"):
                raise ValueError("Unsafe path in model archive")
        package.extractall(folder)
    metadata = {"model": MODEL, "source": URL, "license": "Apache-2.0",
                "sha256": file_sha256(archive), "weights_trained_by_project": False}
    (folder / MODEL / "download_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder / MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    if args.download_model:
        print(install_model(ROOT / ".runtime/speech"))
    for name in ("vosk", "sounddevice", "text2num"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: missing; install requirements-voice.txt")
    print("Model ready:", (ROOT / ".runtime/speech" / MODEL / "am/final.mdl").is_file())


if __name__ == "__main__":
    main()
