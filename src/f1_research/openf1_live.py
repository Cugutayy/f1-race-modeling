"""OpenF1 ingestion with REST bootstrap and MQTT live streaming.

Historical OpenF1 remains useful without credentials. Real-time access is a paid,
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


def _provider_time_key(row: dict[str, Any]) -> tuple[int, Any]:
    """Deterministic provider ordering used by REST capture and replay provenance."""
    raw = row.get("date") or row.get("date_start")
    if raw is None:
        return 1, ""
    text = str(raw)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return 0, parsed.astimezone(UTC)
    except ValueError:
        return 1, text


def _ordered_provider_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=_provider_time_key)


class OpenF1Client:
    def __init__(self, token: str | None = None, timeout_s: float = 30.0):
        self.token = token
        self.timeout_s = timeout_s
        self.session = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=0.75,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
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
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "CugutayyF1RaceIntelligence/0.2",
            },
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
    """Append immutable raw messages and atomically publish state + capture health."""

    def __init__(self, output: Path):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.output / "events.jsonl"
        self.state_path = self.output / "state.json"
        self.manifest_path = self.output / "manifest.json"
        self._raw_hasher = hashlib.sha256()
        self.raw_bytes = 0
        self.count = 0
        self.last_published_count = 0
        self.process_started_at = datetime.now(UTC).isoformat()
        self.stream_status: dict[str, Any] = {
            "connection_state": "bootstrap",
            "connect_count": 0,
            "disconnect_count": 0,
            "last_connect_at": None,
            "last_disconnect_at": None,
            "last_message_at": None,
            "last_error": None,
        }
        if self.raw_path.exists():
            with self.raw_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    self._raw_hasher.update(chunk)
                    self.raw_bytes += len(chunk)
                    self.count += chunk.count(b"\n")
        self.last_published_count = self.count

    def mark_stream(
        self,
        state: str,
        *,
        at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        at = (at or datetime.now(UTC)).astimezone(UTC)
        self.stream_status["connection_state"] = state
        if state == "connected":
            self.stream_status["connect_count"] += 1
            self.stream_status["last_connect_at"] = at.isoformat()
            self.stream_status["last_error"] = None
        elif state in {"disconnected", "reconnecting", "connect_error"}:
            self.stream_status["disconnect_count"] += 1
            self.stream_status["last_disconnect_at"] = at.isoformat()
        if error:
            self.stream_status["last_error"] = str(error)

    def append_many(
        self,
        topic: str,
        payloads: list[dict[str, Any]],
        received_at: datetime,
    ) -> int:
        if not payloads:
            return 0
        stamp = received_at.astimezone(UTC).isoformat()
        encoded_rows = []
        for payload in payloads:
            row = {"topic": topic, "received_at": stamp, "payload": payload}
            encoded_rows.append(
                (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            )
        block = b"".join(encoded_rows)
        with self.raw_path.open("ab") as handle:
            handle.write(block)
        self._raw_hasher.update(block)
        self.raw_bytes += len(block)
        self.count += len(encoded_rows)
        self.stream_status["last_message_at"] = stamp
        return len(encoded_rows)

    def append(self, topic: str, payload: dict[str, Any], received_at: datetime) -> None:
        self.append_many(topic, [payload], received_at)

    def should_publish(self, publish_every: int) -> bool:
        return self.count - self.last_published_count >= max(1, publish_every)

    def publish(self, snapshot: dict[str, Any]) -> None:
        raw = json.dumps(snapshot, indent=2, allow_nan=False).encode()
        state_temp = self.state_path.with_suffix(".tmp")
        state_temp.write_bytes(raw)
        state_temp.replace(self.state_path)
        manifest = {
            "schema_version": 3,
            "provider": "OpenF1",
            "process_started_at": self.process_started_at,
            "captured_rows": self.count,
            "capture_bytes": self.raw_bytes,
            "events_sha256": self._raw_hasher.copy().hexdigest(),
            "state_sha256": hashlib.sha256(raw).hexdigest(),
            "stream": dict(self.stream_status),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        manifest_raw = json.dumps(manifest, indent=2).encode("utf-8")
        manifest_temp = self.manifest_path.with_suffix(".tmp")
        manifest_temp.write_bytes(manifest_raw)
        manifest_temp.replace(self.manifest_path)
        self.last_published_count = self.count


def bootstrap(
    client: OpenF1Client,
    store: RaceStateStore,
    writer: CaptureWriter,
    session_key: int | str = "latest",
) -> dict[str, Any]:
    """REST bootstrap using one canonical order for raw capture and state mutation."""
    topics = (
        "sessions", "drivers", "position", "intervals", "laps", "stints", "pit",
        "weather", "race_control",
    )
    for topic in topics:
        rows = _ordered_provider_rows(client.get(topic, session_key=session_key))
        for row in rows:
            received = datetime.now(UTC)
            writer.append(topic, row, received)
            store.ingest(topic, row, received)
    snapshot = store.snapshot()
    writer.publish(snapshot)
    return snapshot


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


def stream(
    client: OpenF1Client,
    store: RaceStateStore,
    writer: CaptureWriter,
    topics: tuple[str, ...] = DEFAULT_TOPICS,
    publish_every: int = 25,
) -> None:
    """Subscribe to OpenF1 MQTT with reconnect visibility in the capture manifest."""
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
        nonlocal stopped
        received = datetime.now(UTC)
        if int(reason_code) != 0:
            writer.mark_stream("connect_error", at=received, error=f"reason_code={reason_code}")
            writer.publish(store.snapshot(received))
            stopped = True
            return
        writer.mark_stream("connected", at=received)
        for name in topics:
            mqtt_client.subscribe(f"v1/{name}", qos=0)
        writer.publish(store.snapshot(received))

    def on_disconnect(_mqtt_client, _userdata, _disconnect_flags, reason_code, _properties=None):
        received = datetime.now(UTC)
        if stopped:
            writer.mark_stream("disconnected", at=received)
        else:
            writer.mark_stream("reconnecting", at=received, error=f"reason_code={reason_code}")
        writer.publish(store.snapshot(received))

    def on_message(_mqtt_client, _userdata, message):
        received = datetime.now(UTC)
        try:
            decoded = json.loads(message.payload.decode("utf-8"))
            rows = decoded if isinstance(decoded, list) else [decoded]
            topic = message.topic.rsplit("/", 1)[-1]
            valid_rows = [payload for payload in rows if isinstance(payload, dict)]
            writer.append_many(topic, valid_rows, received)
            for payload in valid_rows:
                store.ingest(topic, payload, received)
            if writer.should_publish(publish_every):
                writer.publish(store.snapshot(received))
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            writer.stream_status["last_error"] = f"message_decode: {exc}"
            return

    writer.mark_stream("connecting")
    writer.publish(store.snapshot())
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.username_pw_set(os.environ.get("OPENF1_USERNAME", "token"), client.token)
    mqtt_client.tls_set()
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=45)
    mqtt_client.loop_start()
    try:
        while not stopped:
            time.sleep(0.25)
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        if writer.stream_status["connection_state"] != "connect_error":
            writer.mark_stream("stopped")
        writer.publish(store.snapshot())


def run_capture(
    output: Path,
    session_key: int | str = "latest",
    no_stream: bool = False,
) -> dict[str, Any]:
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
    print(json.dumps({
        "session_key": result.get("session_key"),
        "drivers": len(result.get("drivers", [])),
        "updated_at": result.get("updated_at"),
    }, indent=2))


if __name__ == "__main__":
    main()
