"""OpenF1 ingestion with REST bootstrap and MQTT live streaming.

Historical OpenF1 remains useful without credentials.  Real-time access is a paid,
authenticated provider feature; this module never fabricates a live connection when
credentials or the optional MQTT dependency are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .live_state import RaceStateStore

API = "https://api.openf1.org/v1/"
TOKEN_URL = "https://api.openf1.org/token"
MQTT_HOST = "mqtt.openf1.org"
MQTT_PORT = 8883
DEFAULT_TOPICS = (
    "drivers", "position", "intervals", "laps", "stints", "pit", "car_data",
    "location", "weather", "race_control",
)


class OpenF1Client:
    def __init__(self, token: str | None = None, timeout_s: float = 30.0):
        self.token = token
        self.timeout_s = timeout_s
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.75,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET", "POST"])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": "CugutayyF1RaceIntelligence/0.2"})
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    @classmethod
    def from_env(cls) -> "OpenF1Client":
        token = os.environ.get("OPENF1_TOKEN")
        if not token and os.environ.get("OPENF1_USERNAME") and os.environ.get("OPENF1_PASSWORD"):
            token = cls.obtain_token(os.environ["OPENF1_USERNAME"], os.environ["OPENF1_PASSWORD"])
        return cls(token)

    @staticmethod
    def obtain_token(username: str, password: str) -> str:
        response = requests.post(
            TOKEN_URL,
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "CugutayyF1RaceIntelligence/0.2"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise ValueError("OpenF1 token response did not contain access_token")
        return token

    def get(self, endpoint: str, **filters: Any) -> list[dict[str, Any]]:
        query: list[tuple[str, Any]] = []
        for key, value in filters.items():
            if value is not None:
                query.append((key, value))
        url = API + endpoint
        if query:
            url += "?" + urlencode(query)
        response = self.session.get(url, timeout=self.timeout_s)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise ValueError(f"Unexpected OpenF1 response for {endpoint}")
        return payload


class CaptureWriter:
    """Append immutable raw messages and atomically publish the latest state."""

    def __init__(self, output: Path):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.output / "events.jsonl"
        self.state_path = self.output / "state.json"
        self.manifest_path = self.output / "manifest.json"
        self.count = 0

    def append(self, topic: str, payload: dict[str, Any], received_at: datetime) -> None:
        row = {"topic": topic, "received_at": received_at.astimezone(UTC).isoformat(), "payload": payload}
        line = json.dumps(row, separators=(",", ":"), allow_nan=False)
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.count += 1

    def publish(self, snapshot: dict[str, Any]) -> None:
        raw = json.dumps(snapshot, indent=2, allow_nan=False).encode()
        temp = self.state_path.with_suffix(".tmp")
        temp.write_bytes(raw)
        temp.replace(self.state_path)
        digest = hashlib.sha256(self.raw_path.read_bytes()).hexdigest() if self.raw_path.exists() else None
        manifest = {
            "schema_version": 1,
            "provider": "OpenF1",
            "captured_rows": self.count,
            "events_sha256": digest,
            "state_sha256": hashlib.sha256(raw).hexdigest(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def bootstrap(client: OpenF1Client, store: RaceStateStore, writer: CaptureWriter,
              session_key: int | str = "latest") -> dict[str, Any]:
    """REST bootstrap. Car/location history is intentionally not bulk-downloaded."""
    topics = ("sessions", "drivers", "position", "intervals", "laps", "stints", "pit", "weather", "race_control")
    for topic in topics:
        rows = client.get(topic, session_key=session_key)
        # Historical endpoints can be large; merging is event-time ordered inside the store.
        for row in rows:
            now = datetime.now(UTC)
            writer.append(topic, row, now)
        store.ingest_many(topic, rows)
    writer.publish(store.snapshot())
    return store.snapshot()


def replay_jsonl(path: Path, output: Path | None = None) -> dict[str, Any]:
    """Deterministically reconstruct a saved capture without network access."""
    store = RaceStateStore()
    writer = CaptureWriter(output) if output else None
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("payload"), dict):
                raise ValueError(f"Invalid capture line {line_number}")
            received = datetime.fromisoformat(row["received_at"].replace("Z", "+00:00"))
            store.ingest(str(row["topic"]), row["payload"], received)
            if writer:
                writer.append(str(row["topic"]), row["payload"], received)
    snapshot = store.snapshot()
    if writer:
        writer.publish(snapshot)
    return snapshot


def stream(client: OpenF1Client, store: RaceStateStore, writer: CaptureWriter,
           topics: tuple[str, ...] = DEFAULT_TOPICS, publish_every: int = 25) -> None:
    """Subscribe to OpenF1 MQTT. Requires provider live entitlement and paho-mqtt."""
    if not client.token:
        raise RuntimeError("Live streaming requires OPENF1_TOKEN or OPENF1_USERNAME/OPENF1_PASSWORD")
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("Install the live extra: pip install -e '.[live]'") from exc

    stopped = False

    def stop(*_args):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def on_connect(mqtt_client, _userdata, _flags, reason_code, _properties=None):
        if int(reason_code) != 0:
            raise RuntimeError(f"OpenF1 MQTT connection failed: {reason_code}")
        for name in topics:
            mqtt_client.subscribe(f"v1/{name}", qos=0)

    def on_message(_mqtt_client, _userdata, message):
        nonlocal stopped
        received = datetime.now(UTC)
        try:
            decoded = json.loads(message.payload.decode("utf-8"))
            rows = decoded if isinstance(decoded, list) else [decoded]
            topic = message.topic.rsplit("/", 1)[-1]
            for payload in rows:
                if not isinstance(payload, dict):
                    continue
                writer.append(topic, payload, received)
                store.ingest(topic, payload, received)
            if writer.count % max(1, publish_every) == 0:
                writer.publish(store.snapshot(received))
        except (UnicodeError, json.JSONDecodeError, ValueError):
            # Malformed provider messages are preserved nowhere and never mutate state.
            return

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.username_pw_set(os.environ.get("OPENF1_USERNAME", "token"), client.token)
    mqtt_client.tls_set()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=45)
    mqtt_client.loop_start()
    try:
        while not stopped:
            time.sleep(0.25)
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        writer.publish(store.snapshot())


def run_capture(output: Path, session_key: int | str = "latest", no_stream: bool = False) -> dict[str, Any]:
    client = OpenF1Client.from_env()
    parsed_session = int(session_key) if str(session_key).isdigit() else str(session_key)
    store = RaceStateStore(None if parsed_session == "latest" else int(parsed_session))
    writer = CaptureWriter(output)
    snapshot = bootstrap(client, store, writer, parsed_session)
    if not no_stream:
        stream(client, store, writer)
        snapshot = store.snapshot()
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture or replay OpenF1 event-time state")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--output", type=Path, default=Path("reports/local/live"))
    capture.add_argument("--session-key", default="latest")
    capture.add_argument("--no-stream", action="store_true", help="REST bootstrap only")
    replay = sub.add_parser("replay")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "capture":
        result = run_capture(args.output, args.session_key, args.no_stream)
    else:
        result = replay_jsonl(args.input, args.output)
    print(json.dumps({"session_key": result.get("session_key"),
                      "drivers": len(result.get("drivers", [])),
                      "updated_at": result.get("updated_at")}, indent=2))


if __name__ == "__main__":
    main()
