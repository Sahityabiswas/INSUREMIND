"""End-to-end runner: data -> NLP -> supervised -> PPO -> baselines -> ablations -> report."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import load_config, set_seed
from src import data_pipeline as DP, nlp, agents as AG, evaluate as EV

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--quick", action="store_true", help="Small smoke run; outputs are not full research results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluation = load_config("configs/evaluation.yaml")["evaluation"]
    data = cfg["dataset"]
    if args.quick:
        data["n_conversations"] = 100
        cfg["ppo"]["total_timesteps"] = 256
        evaluation.update(episodes=10, ablation_episodes=10, seeds=[cfg["seed"]])
    set_seed(cfg.get("seed", 42))
    print("== data =="); DP.write_external_stubs()
    rows = DP.build_dataset(data["n_conversations"], cfg["seed"])
    errors = DP.validate(rows)
    if errors:
        raise ValueError(errors[:10])
    print(DP.save_and_split(rows, cfg["seed"], data["train_ratio"], data["val_ratio"], data["n_profiles"]))
    print("== nlp =="); print(nlp_metrics := nlp.train_all())
    print("== supervised =="); AG.train_supervised()
    from src import ppo_numpy as PN
    for i, seed in enumerate(evaluation["seeds"]):
        print(f"== ppo seed {seed} ==", flush=True)
        model = AG.train_ppo(cfg["reward"], cfg["ppo"]["total_timesteps"], seed, cfg["ppo"],
                             f"ppo_seed_{seed}", {"max_turns": cfg["environment"]["max_turns"]})
        if i == 0:
            PN.save(model, os.path.join(AG.ROOT, "results", "checkpoints", "ppo.npz"))
    print("== eval ==", flush=True)
    comp = EV.compare_all(cfg["reward"], evaluation["episodes"], seeds=evaluation["seeds"],
                          max_turns=cfg["environment"]["max_turns"])
    print("== ablations ==", flush=True)
    abl = EV.ablations(cfg["reward"], evaluation["ablation_episodes"], seeds=evaluation["seeds"],
                       ppo_config=cfg["ppo"], max_turns=cfg["environment"]["max_turns"])
    with open(os.path.join(AG.ROOT, "results", "metrics", "run_config.json"), "w") as f:
        json.dump({"config": cfg, "evaluation": evaluation, "quick": args.quick}, f, indent=2)
    print(EV.write_report(nlp_metrics, comp, abl))

if __name__ == "__main__":
    main()
