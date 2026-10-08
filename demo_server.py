"""Local demonstration UI backed by the existing conversation engine."""
import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import importlib.util
import mimetypes
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
import uuid

from src.common import NEEDS, PROFILES
from src.conversation import ConversationSession
from src.generation import OllamaClient
from src.state_products import PRODUCTS, eligible
from src.voice_runtime import voice_artifacts, voice_conversation

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_MESSAGE = 2000
SESSION_TTL = 3600
MAX_AUDIO_BYTES = 16000 * 2 * 30 + 4096


class APIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_settings(data):
    settings = {"age": data.get("age", 35), "budget": data.get("budget", "low"),
                "profile": data.get("profile", "FAMILY_ORIENTED"),
                "policy": data.get("policy", "ppo"), "generator": data.get("generator", "hybrid"),
                "pipeline": data.get("pipeline", "text")}
    if type(settings["age"]) is not int or not 18 <= settings["age"] <= 100:
        raise APIError(400, "Buyer age must be a whole number between 18 and 100.")
    options = {"budget": ("low", "mid", "high", "unknown"), "profile": PROFILES,
               "policy": ("ppo", "rule"), "generator": ("ollama", "template", "hybrid"),
               "pipeline": ("text", "voice")}
    for key, values in options.items():
        if settings[key] not in values:
            raise APIError(400, f"Invalid {key}.")
    return settings


@dataclass
class DemoSession:
    settings: dict
    engine: ConversationSession
    created: str = field(default_factory=now)
    touched: float = field(default_factory=time.monotonic)
    history: list = field(default_factory=list)
    requests: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self, identifier):
        return {"id": identifier, "settings": self.settings, "created": self.created,
                "closed": self.engine.closed, "history": self.history}


class SessionStore:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.speech_lock = threading.Lock()
        self.asr = None

    def create(self, data):
        settings = validate_settings(data)
        arguments = {key: value for key, value in settings.items() if key != "pipeline"}
        try:
            engine = voice_conversation(**arguments) if settings["pipeline"] == "voice" else ConversationSession(**arguments)
        except (RuntimeError, OSError, ValueError) as exc:
            raise APIError(503, f"Agent artifacts unavailable: {exc}") from exc
        if settings["generator"] in ("ollama", "hybrid"):
            engine.generator.client = OllamaClient(model=os.environ.get("INSURANCE_LLM_MODEL", "llama3.2:3b"),
                                                   profile="voice" if settings["generator"] == "hybrid" else "standard")
        record = DemoSession(settings, engine)
        with self.lock:
            expired = [key for key, value in self.sessions.items()
                       if time.monotonic() - value.touched > SESSION_TTL and not value.lock.locked()]
            for key in expired:
                del self.sessions[key]
            if len(self.sessions) >= 100:
                raise APIError(503, "Too many active sessions. Restart the demo server to clear them.")
            identifier = uuid.uuid4().hex
            self.sessions[identifier] = record
        return record.snapshot(identifier)

    def get(self, identifier):
        with self.lock:
            record = self.sessions.get(identifier)
            if record is None or time.monotonic() - record.touched > SESSION_TTL:
                raise APIError(404, "Session expired. Start a new conversation.")
            record.touched = time.monotonic()
        return record

    def reply(self, identifier, data):
        text, request_id = data.get("message"), data.get("request_id")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE:
            raise APIError(400, f"Message must contain 1 to {MAX_MESSAGE} characters.")
        transcript_id = data.get("transcript_request_id")
        if transcript_id is not None and (not isinstance(transcript_id, str) or not 1 <= len(transcript_id) <= 80):
            raise APIError(400, "Invalid transcript reference.")
        fingerprint = "text:" + hashlib.sha256(json.dumps([text.strip(), transcript_id]).encode()).hexdigest()
        with self.turn_record(identifier, request_id, fingerprint) as (record, previous):
            if previous is not None:
                return previous
            voice = None
            if transcript_id is not None:
                draft = record.requests.get(transcript_id, {}).get("result", {})
                if not draft.get("review") or not draft.get("transcript"):
                    raise APIError(400, "That transcript is unavailable in this session. Please record again.")
                voice = {"transcript": draft["transcript"], "reviewed": True,
                         "corrected": text.strip() != draft["transcript"]["normalized_text"]}
            started = time.monotonic()
            result = record.engine.reply(text.strip())
            entry = self.append_entry(record, text.strip(), result, request_id, started, voice)
            self.remember(record, request_id, fingerprint, entry)
            return entry

    @contextmanager
    def turn_record(self, identifier, request_id, fingerprint):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 80:
            raise APIError(400, "A request identifier is required.")
        record = self.get(identifier)
        if not record.lock.acquire(blocking=False):
            raise APIError(409, "A response is still being generated for this session.")
        try:
            if request_id in record.requests:
                previous = record.requests[request_id]
                if previous["fingerprint"] != fingerprint:
                    raise APIError(409, "This request identifier was already used for another input.")
                yield record, previous["result"]
                return
            if record.engine.closed:
                raise APIError(409, "This conversation has ended. Start a new session.")
            if len(record.history) >= 40:
                raise APIError(409, "This demo session reached 40 turns. Start a new session.")
            if len(record.requests) >= 80:
                raise APIError(409, "This session reached its input-attempt limit. Start a new session.")
            yield record, None
        finally:
            record.lock.release()

    @staticmethod
    def append_entry(record, text, response, request_id, started, voice=None):
        entry = {"turn": len(record.history) + 1, "customer": text, "response": response,
                 "request_id": request_id, "timestamp": now(), "input_mode": "voice" if voice else "text",
                 "duration_ms": round((time.monotonic() - started) * 1000)}
        if voice:
            entry["voice"] = voice
        record.history.append(entry)
        return entry

    @staticmethod
    def remember(record, request_id, fingerprint, result):
        record.requests[request_id] = {"fingerprint": fingerprint, "result": result}
        record.touched = time.monotonic()

    def reply_audio(self, identifier, audio, request_id, review=False):
        from src.speech import SpeechError, VoskASR, VoiceSession, read_wav, recognize_audio
        fingerprint = ("preview:" if review else "audio:") + hashlib.sha256(audio).hexdigest()
        with self.turn_record(identifier, request_id, fingerprint) as (record, previous):
            if previous is not None:
                return previous
            if record.settings["pipeline"] != "voice":
                raise APIError(409, "Select the voice-trained pipeline and apply the profile first.")
            if not self.speech_lock.acquire(blocking=False):
                raise APIError(503, "Local speech processing is busy. Please retry.")
            try:
                config, _, _, _ = voice_artifacts()
                folder = ROOT / ".runtime/speech"
                folder.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    incoming, outgoing = Path(tmp) / "buyer.wav", Path(tmp) / "agent.wav"
                    incoming.write_bytes(audio)
                    # Reject bad uploads before loading speech models or advancing the conversation.
                    try:
                        read_wav(incoming, config["asr"]["max_audio_seconds"])
                    except SpeechError as exc:
                        raise APIError(400, str(exc)) from exc
                    if self.asr is None:
                        self.asr = VoskASR(ROOT / config["asr"]["model_path"], config["asr"]["max_audio_seconds"])
                    started = time.monotonic()
                    if review:
                        result = recognize_audio(self.asr, incoming, config["asr"]["min_confidence"])
                        result.update(review=True, request_id=request_id, entry=None)
                        self.remember(record, request_id, fingerprint, result)
                        return result
                    voice = VoiceSession(record.engine, self.asr, None, config["asr"]["min_confidence"])
                    result = voice.reply_audio(incoming, outgoing)
                result.pop("audio_path", None)
                entry = None
                if result["accepted"]:
                    entry = self.append_entry(record, result["transcript"]["normalized_text"], result["response"],
                                              request_id, started,
                                              {"transcript": result["transcript"], "timing": result["timing"],
                                               "spoken_text": result["spoken_text"]})
                result["entry"] = entry
                self.remember(record, request_id, fingerprint, result)
                return result
            except (SpeechError, ImportError) as exc:
                raise APIError(503, f"Speech recognition unavailable: {exc}") from exc
            finally:
                self.speech_lock.release()

    def speak(self, identifier, turn):
        from src.speech import SpeechError, WindowsTTS
        record = self.get(identifier)
        with record.lock:
            if type(turn) is not int or not 1 <= turn <= len(record.history):
                raise APIError(400, "Choose an existing response turn.")
            response = record.history[turn - 1]["response"]
            text = response["text"] if response["validation"]["ok"] else "I cannot verify that response. Please use the text interface for clarification."
        if not self.speech_lock.acquire(blocking=False):
            raise APIError(503, "Local speech processing is busy. Please retry playback.")
        try:
            config, _, _, _ = voice_artifacts()
            folder = ROOT / ".runtime/speech"
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=folder) as tmp:
                path = Path(tmp) / "agent.wav"
                WindowsTTS(config["tts"]["voice"], config["tts"]["rate"]).synthesize(text, path)
                return path.read_bytes()
        except (SpeechError, OSError) as exc:
            raise APIError(503, f"Speech output unavailable: {exc}") from exc
        finally:
            self.speech_lock.release()


def runtime_status():
    config, output, checkpoint, nlp_path = voice_artifacts()
    asr_ready = (ROOT / config["asr"]["model_path"] / "am/final.mdl").is_file()
    dependencies = all(importlib.util.find_spec(name) is not None for name in ("vosk", "text_to_num"))
    return {"ppo_ready": (ROOT / "results/checkpoints/ppo.npz").is_file(),
            "voice": {"ppo_ready": checkpoint.is_file(), "nlp_ready": nlp_path.is_file(),
                      "asr_ready": asr_ready and dependencies, "tts_available": os.name == "nt",
                      "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                      "results_dir": output.relative_to(ROOT).as_posix(),
                      "environment": config["training"]["environment_version"],
                      "max_audio_seconds": config["asr"]["max_audio_seconds"],
                      "min_confidence": config["asr"]["min_confidence"]}}


def recorded_results(pipeline):
    if pipeline == "text":
        path = ROOT / "results/metrics/rl_comparison.csv"
        if not path.is_file():
            raise APIError(404, "No text evaluation found. Run python run_all.py first.")
        with path.open(newline="", encoding="utf-8") as source:
            metrics = [{key: value if key == "agent" else float(value) for key, value in row.items()}
                       for row in csv.DictReader(source)]
        return {"pipeline": pipeline, "metrics": metrics, "source": path.relative_to(ROOT).as_posix()}
    if pipeline != "voice":
        raise APIError(400, "Invalid evaluation pipeline.")
    _, output, _, _ = voice_artifacts()
    evaluation = output / "evaluation.json"
    provenance = output / "evaluation_provenance.json"
    if not evaluation.is_file() or not provenance.is_file():
        raise APIError(404, "No voice evaluation found. Run python train_voice.py first.")
    data = json.loads(evaluation.read_text(encoding="utf-8"))
    training = json.loads(provenance.read_text(encoding="utf-8"))["config"]["training"]
    speech = output / "speech_metrics.json"
    return {"pipeline": pipeline, "aggregate": data["aggregate"], "training": training,
            "source": evaluation.relative_to(ROOT).as_posix(),
            "speech": json.loads(speech.read_text(encoding="utf-8")) if speech.is_file() else None}


def model_status():
    model = os.environ.get("INSURANCE_LLM_MODEL", "llama3.2:3b")
    try:
        with urlopen("http://localhost:11434/api/tags", timeout=2) as response:
            names = [item["name"] for item in json.load(response).get("models", [])]
        return {"online": True, "ready": model in names, "model": model, "models": names}
    except (OSError, ValueError, KeyError, URLError):
        return {"online": False, "ready": False, "model": model, "models": []}


def start_local_ollama():
    executable = ROOT / ".runtime" / "ollama" / "ollama.exe"
    if model_status()["online"] or not executable.is_file():
        return
    runtime = ROOT / ".runtime"
    for folder in ("models", "tmp"):
        (runtime / folder).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OLLAMA_MODELS=str(runtime / "models"), TEMP=str(runtime / "tmp"),
                       TMP=str(runtime / "tmp"), TMPDIR=str(runtime / "tmp"))
    with (runtime / "ollama-server.log").open("a", encoding="utf-8") as log:
        subprocess.Popen([str(executable), "serve"], stdout=log, stderr=subprocess.STDOUT, env=environment,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, DemoHandler)
        self.store = SessionStore()


class DemoHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send_content(self, status, content, kind="application/json; charset=utf-8"):
        if kind.startswith("application/json"):
            content = json.dumps(content, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Permissions-Policy", "microphone=(self)")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; media-src 'self' blob:; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(content)

    def validate_origin(self):
        port = self.server.server_port
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        if self.headers.get("Host") not in hosts:
            raise APIError(403, "Only local demo requests are allowed.")
        origin = self.headers.get("Origin")
        if origin and origin not in [f"http://{host}" for host in hosts]:
            raise APIError(403, "Cross-origin requests are not allowed.")

    def read_json(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise APIError(415, "Requests must use application/json.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise APIError(413, "Request body is too large or empty.")
            raw = self.rfile.read(length)
            self.body_consumed = True
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise APIError(400, "Expected a JSON object.")
            return data
        except (ValueError, UnicodeError):
            raise APIError(400, "Invalid JSON request.") from None

    def read_audio(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "audio/wav":
            raise APIError(415, "Audio requests must use audio/wav.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid audio length.") from None
        if not 44 < length <= MAX_AUDIO_BYTES:
            raise APIError(413, "Audio must be a PCM WAV of at most 30 seconds.")
        audio = self.rfile.read(length)
        self.body_consumed = True
        if len(audio) != length:
            raise APIError(400, "Incomplete audio upload.")
        return audio

    def dispatch(self, method):
        self.body_consumed = False
        try:
            self.validate_origin()
            parsed = urlparse(self.path)
            path = parsed.path
            if method == "GET":
                if path == "/api/status":
                    self.send_content(200, {"ollama": model_status(), **runtime_status(),
                                           "profiles": PROFILES, "needs": NEEDS, "max_message": MAX_MESSAGE})
                elif path == "/api/products":
                    query = parse_qs(parsed.query)
                    try:
                        age = int(query.get("age", ["35"])[0])
                    except ValueError:
                        raise APIError(400, "Invalid age.") from None
                    budget = query.get("budget", ["low"])[0]
                    need = query.get("need", ["GENERAL_FINANCIAL_PROTECTION"])[0]
                    validate_settings({"age": age, "budget": budget})
                    if need not in NEEDS:
                        raise APIError(400, "Invalid need.")
                    self.send_content(200, {"products": [{**product, "eligible": eligible(product, need, budget, age)}
                                                           for product in PRODUCTS]})
                elif path == "/api/results":
                    self.send_content(200, recorded_results(parse_qs(parsed.query).get("pipeline", ["text"])[0]))
                elif path.startswith("/api/sessions/"):
                    identifier = path.removeprefix("/api/sessions/")
                    record = self.server.store.get(identifier)
                    with record.lock:
                        self.send_content(200, record.snapshot(identifier))
                else:
                    assets = {"/": WEB / "index.html", "/styles.css": WEB / "styles.css", "/app.js": WEB / "app.js",
                              "/audio.js": WEB / "audio.js", "/audio-worklet.js": WEB / "audio-worklet.js",
                              "/vendor/lucide.min.js": WEB / "vendor/lucide.min.js",
                              "/assets/reward-curve.png": ROOT / "results/metrics/ppo_seed_42_reward_curve.png"}
                    if path == "/favicon.ico":
                        self.send_content(204, b"", "image/x-icon")
                    elif path in assets and assets[path].is_file():
                        file = assets[path]
                        kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                        if file.suffix == ".js":
                            kind = "text/javascript"
                        self.send_content(200, file.read_bytes(), kind)
                    else:
                        raise APIError(404, "Not found.")
            elif method == "POST":
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "audio":
                    audio = self.read_audio()
                    review = parse_qs(parsed.query).get("review", ["0"])[0] == "1"
                    self.send_content(200, self.server.store.reply_audio(parts[2], audio, self.headers.get("X-Request-ID"), review))
                    return
                data = self.read_json()
                if path == "/api/sessions":
                    self.send_content(201, self.server.store.create(data))
                elif len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "messages":
                    self.send_content(200, self.server.store.reply(parts[2], data))
                elif len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "speech":
                    self.send_content(200, self.server.store.speak(parts[2], data.get("turn")), "audio/wav")
                else:
                    raise APIError(404, "Not found.")
        except APIError as error:
            # Windows can reset the connection if a rejected request has an unread body.
            if method == "POST" and not self.body_consumed:
                self.close_connection = True
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if 0 < length <= 16384:
                        self.rfile.read(length)
                except (OSError, ValueError):
                    pass
            self.send_content(error.status, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            print(f"Demo request failed: {type(error).__name__}: {error}", flush=True)
            self.send_content(500, {"error": "The agent could not complete this request. Check the demo server log."})

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-start-ollama", action="store_true")
    args = parser.parse_args()
    if not args.no_start_ollama:
        start_local_ollama()
    try:
        server = DemoServer(("127.0.0.1", args.port))
    except OSError as error:
        parser.error(f"Cannot use port {args.port}: {error}. Try --port {args.port + 1}.")
    print(f"Insurance demo: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
