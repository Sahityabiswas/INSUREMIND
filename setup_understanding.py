"""Download pinned, data-only model assets to this project's D-drive runtime."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "minilm": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "license": "apache-2.0",
        "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.txt", "README.md"],
    },
    "roberta": {
        "repo": "SamLowe/roberta-base-go_emotions-onnx",
        "revision": "90ee0c1c4796d370e68968687b8ba51fc11224f4", "license": "mit",
        "files": ["config.json", "tokenizer.json", "onnx/model_quantized.onnx", "README.md"],
    },
    "acoustic": {
        "repo": "onnx-community/wav2vec2-base-superb-er-ONNX",
        "revision": "e2191e9e8b692c19c7d96e2a1d789ab1df5b7259", "license": "apache-2.0",
        "files": ["config.json", "preprocessor_config.json", "onnx/model.onnx", "README.md"],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(ASSETS), default=list(ASSETS))
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    from src.speech import file_sha256
    for name in args.models:
        spec = ASSETS[name]
        target = ROOT / ".runtime/understanding" / name
        hashes = {}
        for filename in spec["files"]:
            print(f"Downloading {name}/{filename}", flush=True)
            path = hf_hub_download(spec["repo"], filename, revision=spec["revision"], local_dir=target)
            hashes[filename] = file_sha256(path)
        (target / "provenance.json").write_text(json.dumps({**spec, "sha256": hashes}, indent=2), encoding="utf-8")
        print(f"Ready: {target}", flush=True)


if __name__ == "__main__":
    main()
