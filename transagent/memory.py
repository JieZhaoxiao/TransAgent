"""Working, episodic, and long-term memory without target-model feedback."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from .schemas import AttackState, TransformProgram


def _state_vector(state: AttackState) -> list[float]:
    values = state.model_dump()
    result = [state.step / max(1, state.total_steps - 1)]
    for name, value in values.items():
        if name not in {"step", "total_steps"} and isinstance(value, (float, int)):
            result.append(float(value))
    return result


def _cosine(first: list[float], second: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(first, second))
    first_norm = sum(value * value for value in first) ** 0.5
    second_norm = sum(value * value for value in second) ** 0.5
    return numerator / max(first_norm * second_norm, 1e-12)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = json.loads(json.dumps(record, ensure_ascii=True, allow_nan=False))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitized, ensure_ascii=True, sort_keys=True) + "\n")


class HierarchicalMemory:
    def __init__(self, database: str | Path, event_path: str | Path, working_size: int = 64):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.events = Path(event_path)
        self.working = deque(maxlen=working_size)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.database, timeout=30, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodic_memory (
              id INTEGER PRIMARY KEY, episode_id TEXT, step INTEGER, state_json TEXT,
              program_json TEXT, immediate_reward REAL, delayed_reward REAL,
              success_reason TEXT, failure_reason TEXT, compute_cost REAL, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS long_term_memory (
              state_pattern TEXT, program_id TEXT, program_json TEXT, visits INTEGER,
              mean_reward REAL, reward_m2 REAL, success_count INTEGER, phases TEXT,
              image_features TEXT, updated_at TEXT, PRIMARY KEY(state_pattern, program_id)
            );
            """
        )
        self._connection.commit()

    def add_working(self, record: dict[str, Any]) -> None:
        self.working.append(record)

    def retrieve(self, state: AttackState, limit: int = 7) -> list[dict[str, Any]]:
        phase = state.phase
        rows = self._connection.execute(
            "SELECT state_pattern, program_id, program_json, visits, mean_reward, "
            "reward_m2, success_count,image_features FROM long_term_memory WHERE phases LIKE ?",
            (f"%{phase}%",)
        ).fetchall()
        current = _state_vector(state)
        results = []
        for row in rows:
            features = json.loads(row[7])
            historical = features.get("state_vector", [])
            similarity = _cosine(current, historical) if historical else 0.0
            results.append({"state_pattern": row[0], "program_id": row[1],
                "program": json.loads(row[2]), "visits": row[3], "mean_reward": row[4],
                "reward_variance": row[5] / max(1, row[3] - 1),
                "success_rate": row[6] / max(1, row[3]), "state_similarity": similarity})
        results.sort(key=lambda item: (-item["state_similarity"], -item["mean_reward"], -item["visits"]))
        results = results[:limit]
        append_jsonl(self.events, {"event": "memory_retrieve", "phase": phase,
                                  "limit": limit, "match_count": len(results),
                                  "hit": int(bool(results)),
                                  "timestamp": datetime.now(timezone.utc).isoformat()})
        return results

    def store(
        self,
        *,
        episode_id: str,
        step: int,
        state: AttackState,
        program: TransformProgram,
        immediate_reward: float,
        delayed_reward: float,
        cost: float,
        reason: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        state_pattern = f"{state.phase}:hf{round(state.high_frequency_energy, 1)}:edge{round(state.edge_density, 1)}"
        success = int(immediate_reward > 0)
        with self._lock:
            self._connection.execute(
                "INSERT INTO episodic_memory(episode_id,step,state_json,program_json,immediate_reward,"
                "delayed_reward,success_reason,failure_reason,compute_cost,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (episode_id, step, state.model_dump_json(), program.model_dump_json(), immediate_reward,
                 delayed_reward, reason if success else "", "" if success else reason, cost, now),
            )
            old = self._connection.execute(
                "SELECT visits,mean_reward,reward_m2,success_count FROM long_term_memory "
                "WHERE state_pattern=? AND program_id=?", (state_pattern, program.program_id)
            ).fetchone()
            if old:
                visits = old[0] + 1
                delta = immediate_reward - old[1]
                mean = old[1] + delta / visits
                m2 = old[2] + delta * (immediate_reward - mean)
                self._connection.execute(
                    "UPDATE long_term_memory SET visits=?,mean_reward=?,reward_m2=?,success_count=?,"
                    "updated_at=? WHERE state_pattern=? AND program_id=?",
                    (visits, mean, m2, old[3] + success, now, state_pattern, program.program_id),
                )
            else:
                features = {"state_vector": _state_vector(state)}
                self._connection.execute(
                    "INSERT INTO long_term_memory VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (state_pattern, program.program_id, program.model_dump_json(), 1, immediate_reward,
                     0.0, success, json.dumps(program.phases), json.dumps(features), now),
                )
            self._connection.commit()
        append_jsonl(self.events, {"event": "memory_store", "episode_id": episode_id,
                                  "step": step, "program_id": program.program_id,
                                  "reward": immediate_reward, "timestamp": now})

    def finalize_episode(self, episode_id: str, gamma: float = 0.9) -> None:
        rows = self._connection.execute(
            "SELECT id,immediate_reward FROM episodic_memory WHERE episode_id=? ORDER BY step", (episode_id,)
        ).fetchall()
        delayed = 0.0
        updates = []
        for row_id, reward in reversed(rows):
            delayed = float(reward) + gamma * delayed
            updates.append((delayed - float(reward), row_id))
        with self._lock:
            self._connection.executemany("UPDATE episodic_memory SET delayed_reward=? WHERE id=?", updates)
            self._connection.commit()
        append_jsonl(self.events, {"event": "episode_delayed_rewards_finalized",
                                  "episode_id": episode_id, "records": len(updates), "gamma": gamma})

    def close(self) -> None:
        self._connection.close()
