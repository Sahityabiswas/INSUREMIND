"""Shared voice artifact selection for the CLI and local web application."""
from pathlib import Path

from .common import load_config
from .conversation import ConversationSession

ROOT = Path(__file__).resolve().parents[1]


def voice_artifacts(config_path=None, backend=None):
    from .understanding import selected_backend
    backend = backend or selected_backend()
    default = "configs/voice_transformer.yaml" if backend == "transformer" else "configs/voice.yaml"
    config = load_config(config_path or ROOT / default)
    output = ROOT / config["training"]["output_dir"]
    return config, output, output / "checkpoints/ppo_voice.npz", output / "nlp_models.json"


def voice_conversation(*, config_path=None, checkpoint=None, nlp_model=None, **settings):
    from .voice_training import load_predictor
    from .understanding import selected_backend
    backend = settings.get("nlp_backend") or selected_backend()
    _, _, default_checkpoint, default_nlp = voice_artifacts(config_path, backend)
    checkpoint, nlp_model = Path(checkpoint or default_checkpoint), Path(nlp_model or default_nlp)
    if not nlp_model.is_file() or (settings.get("policy", "ppo") == "ppo" and not checkpoint.is_file()):
        raise RuntimeError("Voice training artifacts missing. Run python train_voice.py first.")
    return ConversationSession(checkpoint=checkpoint, predictor=load_predictor(nlp_model), **settings)
