"""Prepare synthetic speech, adapt NLP, train speech-conditioned PPO, and evaluate."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from src import agents, ppo_numpy
from src.common import ACTIONS, load_config
from src.speech import ROOT, VoskASR, WindowsTTS, file_sha256
from src.state_products import STATE_DIM
from src.voice_training import (SpeechInsuranceEnv, TranscriptChannel, evaluate_policy, load_predictor,
    prepare_corpus, read_jsonl, speech_metrics, train_nlp, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/voice_transformer.yaml"))
    parser.add_argument("--base-config", default=str(ROOT / "configs/base.yaml"))
    parser.add_argument("--reuse-data", action="store_true", help="Use the existing transcribed manifest")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate saved voice checkpoints without training or changing them")
    parser.add_argument("--quick", action="store_true", help="Smoke run in separate *_smoke folders, not a research result")
    args = parser.parse_args()
    cfg, base = load_config(args.config), load_config(args.base_config)
    settings = dict(cfg["training"])
    observable = settings.get("observable_dialogue", False)
    output, dataset = ROOT / settings["output_dir"], ROOT / settings["dataset_dir"]
    if not args.evaluate_only and not args.quick and (output / "run.json").is_file():
        parser.error("Completed experiment exists. Set new output_dir and dataset_dir in a copied config to preserve it.")
    if args.quick:
        output = output.with_name(output.name + "_smoke")
        dataset = dataset.with_name(dataset.name + "_smoke")
        settings.update(seeds=[42], total_timesteps=256, evaluation_episodes=5, nlp_examples_per_split=8)
    output.mkdir(parents=True, exist_ok=True)
    print("Speech outputs:", output, flush=True)
    print("Pretrained ASR/TTS weights stay fixed. Only insurance NLP/PPO are trained.", flush=True)
    if args.reuse_data or args.evaluate_only:
        rows = read_jsonl(dataset / "manifest.jsonl")
        for row in rows:
            if not Path(row["wav"]).is_file() or file_sha256(row["wav"]) != row["audio_sha256"]:
                raise ValueError(f"Missing or modified audio: {row['wav']}")
    else:
        asr = VoskASR(ROOT / cfg["asr"]["model_path"], cfg["asr"]["max_audio_seconds"])
        tts = WindowsTTS(cfg["tts"]["voice"], cfg["tts"]["rate"])
        previous = ROOT / settings["reuse_manifest"] if settings.get("reuse_manifest") else None
        rows = prepare_corpus(dataset, asr, tts, settings["nlp_examples_per_split"], settings["speech_rates"],
                              observable_dialogue=observable, reuse_manifest=previous)
    speech = {split: speech_metrics([r for r in rows if r["split"] == split]) for split in ("train", "val", "test")}
    write_json(output / "speech_metrics.json", speech)
    if args.prepare_only:
        print(json.dumps(speech, indent=2))
        return
    if args.evaluate_only:
        predictor = load_predictor(output / "nlp_models.json")
    elif settings.get("nlp_backend") == "transformer":
        write_json(output / "nlp_models.json", {"version": 2, "backend": "transformer",
                   "model_dir": settings["nlp_model_dir"]})
        predictor = load_predictor(output / "nlp_models.json")
    else:
        predictor, _ = train_nlp(rows, output)
    weights = None
    initial = ROOT / "results/supervised_policy.pkl"
    if initial.exists():
        with initial.open("rb") as handle:
            weights = pickle.load(handle)["W"]
    channels = {split: TranscriptChannel(rows, split, observable) for split in ("train", "test")}
    def environment(split):
        return SpeechInsuranceEnv(channels[split], predictor, base["reward"],
                                  base["environment"]["max_turns"], cfg["asr"]["min_confidence"], observable)
    ppo_settings = dict(base["ppo"])
    ppo_settings.pop("total_timesteps", None)
    ppo_settings["lr"] = ppo_settings.pop("learning_rate")
    summaries, episodes, trajectories, histories = [], [], [], {}
    baseline_path = ROOT / "results/checkpoints/ppo.npz"
    baseline = ppo_numpy.load(str(baseline_path), STATE_DIM, len(ACTIONS)) if baseline_path.exists() else None
    prior_path = ROOT / "results/voice/checkpoints/ppo_voice.npz"
    prior = ppo_numpy.load(str(prior_path), STATE_DIM, len(ACTIONS)) if prior_path.exists() and output.name != "voice" else None
    for index, seed in enumerate(settings["seeds"]):
        path = output / "checkpoints" / f"ppo_voice_seed_{seed}.npz"
        if args.evaluate_only:
            model = ppo_numpy.load(str(path), STATE_DIM, len(ACTIONS))
            histories[str(seed)] = json.loads((output / f"training_seed_{seed}.json").read_text(encoding="utf-8"))
        else:
            print(f"Training voice PPO seed {seed}: {settings['total_timesteps']} steps", flush=True)
            start = time.perf_counter()
            model, mean_reward = ppo_numpy.train(environment("train"), total_steps=settings["total_timesteps"],
                                                seed=seed, init_W=weights, **ppo_settings)
            ppo_numpy.save(model, str(path))
            if index == 0:
                shutil.copyfile(path, output / "checkpoints/ppo_voice.npz")
            histories[str(seed)] = {"mean_train_reward": mean_reward, "history": model.history,
                                   "training_seconds": time.perf_counter() - start}
            write_json(output / f"training_seed_{seed}.json", histories[str(seed)])
        policies = {"voice_ppo": agents.ppo_policy(model), "rule": agents.rule_policy()}
        if baseline is not None:
            policies["text_trained_ppo"] = agents.ppo_policy(baseline)
        if prior is not None:
            policies["prior_voice_ppo"] = agents.ppo_policy(prior)
        for name, policy in policies.items():
            print(f"Evaluating {name}, seed {seed}", flush=True)
            transitions = []
            summary, traces = evaluate_policy(environment("test"), policy, settings["evaluation_episodes"], seed, transitions)
            summaries.append({"agent": name, "seed": seed, **summary})
            episodes.extend({"agent": name, **row} for row in traces)
            trajectories.extend({"agent": name, **row} for row in transitions)
    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["agent"]].append(row)
    aggregate = {name: {metric: {"mean": float(np.mean([r[metric] for r in group])),
                                "seed_sd": float(np.std([r[metric] for r in group], ddof=1)) if len(group) > 1 else None}
                        for metric in ("reward", "conversion", "satisfaction", "violations", "asr_retries", "missed_stop_turns",
                                       "interest_turns", "commitment_attempts", "masked_action_repairs")}
                 for name, group in grouped.items()}
    write_json(output / "evaluation.json", {"per_seed": summaries, "aggregate": aggregate})
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for transition in trajectories:
            handle.write(json.dumps(transition) + "\n")
    run_metadata = {"quick": args.quick, "config": cfg, "effective_training": settings,
        "ppo": ppo_settings, "base_reward": base["reward"], "histories": histories,
        "manifest_sha256": file_sha256(dataset / "manifest.jsonl"),
        "text_checkpoint_sha256": file_sha256(baseline_path) if baseline else None,
        "asr_tts_weights_updated": False, "selected_checkpoint": "first configured seed, not test-selected",
        "nlp_training": "Pre-fine-tuned MiniLM and mapped RoBERTa" if settings.get("nlp_backend") == "transformer" else "Original text training split plus its transcribed synthetic speech subset",
        "ppo_training": "ASR-derived live observation state, simulator reward and transitions"}
    if not args.evaluate_only:
        write_json(output / "run.json", run_metadata)
    write_json(output / "evaluation_provenance.json", {"config": cfg, "base_reward": base["reward"],
        "nlp_bundle_sha256": file_sha256(output / "nlp_models.json"),
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in [Path(__file__), *sorted((ROOT / "src").glob("*.py"))]},
        "manifest_sha256": file_sha256(dataset / "manifest.jsonl"),
        "evaluated_checkpoints": {str(p.relative_to(ROOT)): file_sha256(p) for p in
            [*(output / "checkpoints").glob("ppo_voice_seed_*.npz"), *([baseline_path] if baseline else []), *([prior_path] if prior else [])]}})
    lines = ["# Voice Training and Evaluation", "", f"Run type: {'SMOKE ONLY' if args.quick else 'configured experiment'}.",
             f"Environment: {settings.get('environment_version', 'voice_v1')}. Purchase threshold and reward weights unchanged.",
             "Pretrained Vosk ASR and Windows SAPI TTS; their weights were not updated.", "",
             f"PPO: {settings['total_timesteps']} steps per seed; seeds {settings['seeds']}; "
             f"{settings['evaluation_episodes']} evaluation episodes per seed and policy.", "",
             "## Speech Recognition", "", "| Split | Examples | Word error rate | Character error rate |",
             "| --- | ---: | ---: | ---: |"]
    for split, metrics in speech.items():
        lines.append(f"| {split} | {metrics['examples']} | {metrics['wer']:.2%} | {metrics['cer']:.2%} |")
    lines += ["", "## Policy Comparison", "", "All policies receive the same held-out-rate ASR observations and adapted NLP.",
              "The text-trained checkpoint is evaluated here without modifying it.", "",
              "| Policy | Mean reward | Seed SD | Simulated conversion | Mean violation turns |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, metrics in aggregate.items():
        sd = metrics["reward"]["seed_sd"]
        sd_text = f"{sd:.4f}" if sd is not None else "n/a (one seed)"
        lines.append(f"| {name} | {metrics['reward']['mean']:.4f} | {sd_text} | "
                     f"{metrics['conversion']['mean']:.2%} | {metrics['violations']['mean']:.4f} |")
    lines += ["", "## Limits", "",
              "- Synthetic system-voice audio, not recordings from real buyers. One speaker; held-out speech rates only.",
              "- NLP uses a separate distinct-sentence synthetic fine-tuning corpus; speech simulator templates repeat." if settings.get("nlp_backend") == "transformer" else "- NLP uses original conversation splits. Repeated synthetic utterance templates remain a limitation.",
              "- Simulator utterances repeat across speech splits; this tests channel variation, not unseen language.",
              "- The policy receives live-style defaults for hidden trust/intent and ASR-derived entities, not hidden simulator state.",
              "- Profile and age are supplied enrollment fields; the simulator retains hidden state to compute reward.",
              "- Low-confidence simulator retries advance one turn; live audio retries leave buyer memory unchanged.",
              "- Acoustic emotion is advisory in live inference, not a PPO feature or trained in this experiment.",
              "- No speech-model or Llama fine-tuning was performed.",
              "- Microphone interaction, accent/noise robustness and human sales outcomes require separate testing.",
              "- The original text experiment and its checkpoints are unchanged.", "",
              "- Observable-dialogue mode verbalizes existing simulated objections/readiness. Compare policies within this version;",
              "  do not attribute differences from the original voice report solely to retraining.", "",
              ("Detailed NLP results: `../understanding_v1/evaluation.json`." if settings.get("nlp_backend") == "transformer" else "Detailed NLP results: `nlp_metrics.json`.") + " Per-episode results: `episodes.csv`."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Completed:", output / "report.md", flush=True)


if __name__ == "__main__":
    main()
