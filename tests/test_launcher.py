import sys

import run_llm as launcher


def test_slow_server_start_uses_project_storage(tmp_path, monkeypatch):
    executable = tmp_path / ".runtime" / "ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_llm.py", "--debug"])
    for name in ("OLLAMA_MODELS", "INSURANCE_LLM_MODEL", "INSURANCE_LLM_URL"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("INSURANCE_LLM_TIMEOUT", "240")
    clock = {"seconds": 0}

    def models():
        if clock["seconds"] < 75:
            raise OSError("Hardware detection is still running")
        return {"llama3.2:3b"}

    def sleep(seconds):
        clock["seconds"] += seconds

    started, commands = [], []
    monkeypatch.setattr(launcher, "server_models", models)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock["seconds"])
    monkeypatch.setattr(launcher.time, "sleep", sleep)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: started.append((args, kwargs)))
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, **kwargs: commands.append(command) or 0)

    assert launcher.main() == 0
    assert clock["seconds"] == 75
    environment = started[0][1]["env"]
    assert environment["OLLAMA_MODELS"] == str(tmp_path / ".runtime" / "models")
    for name in ("TEMP", "TMP", "TMPDIR"):
        assert environment[name] == str(tmp_path / ".runtime" / "tmp")
    assert environment["INSURANCE_LLM_TIMEOUT"] == "240"
    assert commands[0][2:4] == ["--generator", "ollama"]
    assert "--debug" in commands[0]


def test_voice_launcher_reuses_installed_model(tmp_path, monkeypatch):
    executable = tmp_path / ".runtime" / "ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_llm.py", "--voice", "--age", "35"])
    monkeypatch.setattr(launcher, "server_models", lambda: {"llama3.2:3b"})
    commands = []
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, **kwargs: commands.append(command) or 0)
    assert launcher.main() == 0
    assert commands[0][1] == str(tmp_path / "voice.py")
    assert commands[0][2:4] == ["--generator", "hybrid"]
