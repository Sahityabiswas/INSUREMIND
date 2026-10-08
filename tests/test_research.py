import json
import numpy as np
import pytest

from src.common import ACTIONS
from src.generation import ResponseGenerator, validate_response
from src.sim_env import InsuranceEnv, action_mask
from src.state_products import STATE_DIM, encode, PRODUCTS, ablate_observation
from src.ppo_numpy import LinearPPO, advantages, train, save, load


def test_gym_contract():
    from gymnasium.utils.env_checker import check_env
    check_env(InsuranceEnv(), skip_render_check=True)
    env = InsuranceEnv(max_turns=1)
    env.reset(seed=7)
    _, reward, terminated, truncated, _ = env.step(0)
    assert np.isfinite(reward) and not terminated and truncated
    with pytest.raises(RuntimeError):
        env.step(0)


def test_rejection_mask_and_eligibility():
    mask = action_mask("DISCOVERY", 5, True, stop_requested=True)
    assert sum(mask) == 1 and mask[ACTIONS.index("RESPECT_REJECTION")]
    assert not action_mask("DISCOVERY", 5, True)[ACTIONS.index("DISCOVER_NEEDS")]
    env = InsuranceEnv()
    _, info = env.reset(seed=1, options={"profile": {"age": 99, "primary_need": "RETIREMENT", "budget": "low"}})
    assert not info["action_mask"][ACTIONS.index("EXPLAIN_PRODUCT")]
    env.step(0)
    _, _, _, _, info = env.step(ACTIONS.index("EXPLAIN_PRODUCT"))
    assert info["unsuitable"] and info["safety_violation"]


def test_gae_keeps_raw_value_targets_and_bootstraps_time_limits():
    adv, returns = advantages([1, 2], [0.5, 0.7], [0.7, 9], [False, True], [False, True], 1, 1)
    assert np.allclose(adv, [2.5, 1.3])
    assert np.allclose(returns, [3, 2])
    _, truncated_return = advantages([1], [0.5], [2], [False], [True], 0.9, 0.95)
    assert np.allclose(truncated_return, [2.8])


def test_masked_sampling_and_checkpoint(tmp_path):
    model = LinearPPO(STATE_DIM, len(ACTIONS), seed=4)
    obs = np.zeros(STATE_DIM)
    mask = [i in (0, 1) for i in range(len(ACTIONS))]
    assert np.allclose(model.probs(obs, mask)[2:], 0)
    sampled = [model.act(obs, mask)[0] for _ in range(40)]
    assert set(sampled) == {0, 1}
    model, avg = train(InsuranceEnv(), total_steps=64, n_steps=32, batch_size=16, seed=5)
    assert len(model.history) == 2 and np.isfinite(avg)
    assert np.isfinite(model.W).all() and np.isfinite(model.vw).all()
    path = str(tmp_path / "model.npz")
    save(model, path)
    assert np.allclose(model.W, load(path, STATE_DIM, len(ACTIONS)).W)


def test_state_defaults_and_ablation():
    vector = encode({"trust": 7, "satisfaction": -2, "purchase_intent": float("nan")})
    assert len(vector) == STATE_DIM and all(0 <= x <= 1 for x in vector)
    altered = ablate_observation(vector, ["emotion", "objection"])
    assert not altered[9:17].any() and not altered[26:37].any() and not altered[80:88].any()


def test_llm_strategy_and_fact_rejection():
    class FakeClient:
        def complete(self, system, payload):
            return {"strategy": "COMMITMENT_REQUEST", "text": "Guaranteed approval with 99L cover and 2 days waiting."}
    result = ResponseGenerator("ollama", FakeClient()).generate("EXPLAIN_PRODUCT", {"need": "FAMILY_HEALTH", "budget": "low"})
    assert result["llm_rejected"] and result["source"] == "template"
    assert "99L" not in result["text"]
    assert not validate_response("Waiting period is 99 days.", [PRODUCTS[0]])["ok"]
    assert validate_response("same", [], ["same"])["repeated"]


def test_session_stop_and_persistent_entities():
    from src.conversation import ConversationSession
    session = ConversationSession("rule", age=35)
    session.reply("I need family health insurance with a low budget.")
    response = session.reply("I already have a 5 lakh policy.")
    assert response["state"]["need"] == "FAMILY_HEALTH"
    response = session.reply("No, I am not interested, please stop.")
    assert response["action"] == "RESPECT_REJECTION" and response["closed"]
    with pytest.raises(RuntimeError):
        session.reply("again")


def test_scenario_group_split_and_derived_data(tmp_path, monkeypatch):
    from src import data_pipeline as dp
    monkeypatch.setattr(dp, "ROOT", str(tmp_path))
    rows = dp.build_dataset(60, 42)
    assert dp.validate(rows) == []
    assert dp.validate([{}])
    stats = dp.save_and_split(rows, n_profiles=25)
    assert stats["profiles"] == 25
    scenarios = []
    for split in ("train", "val", "test"):
        path = tmp_path / "data" / "processed" / "insurance_sales" / f"{split}.jsonl"
        scenarios.append({json.loads(line)["scenario_id"] for line in path.read_text().splitlines()})
    assert not scenarios[0] & scenarios[1] and not scenarios[0] & scenarios[2] and not scenarios[1] & scenarios[2]
    objections = [json.loads(line) for line in (tmp_path / "data" / "processed" / "objections.jsonl").read_text().splitlines()]
    assert objections and all(r["seller_response"] and r["source_type"] == "derived" for r in objections)


def test_episode_metrics_preserve_earlier_events():
    from src.agents import run_policy
    class EventEnv(InsuranceEnv):
        def step(self, action):
            o, r, term, trunc, info = super().step(action)
            info["pressure"] = self.t == 1
            return o, r, term, trunc, info
    env = EventEnv(max_turns=2)
    rows, trajectories = [], []
    result = run_policy(env, lambda o, m, i: next(a for a, ok in enumerate(m) if ok), 1, 42, "test", rows, trajectories)
    assert result["pressure_violations"] == 1
    assert len(trajectories) == 2 and rows[0]["avg_turns"] == 2


def test_weighted_f1_uses_class_support():
    from src.nlp import _scores
    result = _scores(["a"] * 9 + ["b"], ["a"] * 10, ["a", "b"])
    assert result["weighted_f1"] > result["macro_f1"]


def test_external_reader_preserves_provenance_and_rejects_leakage(tmp_path):
    from src.external_data import import_corpus
    raw = tmp_path / "train.tsv"
    raw.write_text("Happy to hear that\t17\tcomment_1\n", encoding="utf-8")
    path = import_corpus(raw, "goemotions", output_dir=tmp_path / "normalized")
    row = json.loads(path.read_text())
    assert row["source_type"] == "public" and row["customer_emotion"] == "POSITIVE"
    assert row["native_labels"] == ["joy"] and row["label_source"] == "derived_mapping"
    with pytest.raises(ValueError, match="another split"):
        import_corpus(raw, "goemotions", "test", output_dir=tmp_path / "normalized")
