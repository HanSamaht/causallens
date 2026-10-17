"""evidence ledger + case archive(SQLite,).

:
  1. —— everyevidence/hypothesis/,alsoin
  2. —— evidence(every), verify
  3. —— hypothesis valid_from/valid_to:"findingXinv3,v4"

:MLflow run-> schema,EU AI Act IV,W3C PROV,
Graphiti model;"historyhypothesisasinvestigationprior"asprojecthas.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
  id INTEGER PRIMARY KEY,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(model_name, model_version)
);
CREATE TABLE IF NOT EXISTS evidence(
  id INTEGER PRIMARY KEY,
  case_id INTEGER NOT NULL,
  ts TEXT NOT NULL,
  tool TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  prev_hash TEXT,
  hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence(case_id);
CREATE TABLE IF NOT EXISTS hypotheses(
  id INTEGER PRIMARY KEY,
  case_id INTEGER NOT NULL,
  statement TEXT NOT NULL,
  verdict TEXT NOT NULL,
  posterior REAL,
  valid_from TEXT NOT NULL,
  valid_to TEXT
);
CREATE TABLE IF NOT EXISTS corrections(
  id INTEGER PRIMARY KEY,
  case_id INTEGER NOT NULL,
  edge TEXT NOT NULL,
  operation TEXT NOT NULL,
  note TEXT,
  verification_json TEXT,
  ts TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: str = "mic_ledger.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ---------------- ----------------
    def open_case(self, model_name: str, model_version: str = "v1") -> int:
        row = self.conn.execute(
            "SELECT id FROM cases WHERE model_name=? AND model_version=?",
            (model_name, model_version),
        ).fetchone()
        if row:
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO cases(model_name, model_version, created_at) VALUES(?,?,?)",
            (model_name, model_version, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ---------------- evidence() ----------------
    def append_evidence(self, case_id: int, tool: str, payload: dict) -> dict:
        last = self.conn.execute(
            "SELECT hash FROM evidence WHERE case_id=? ORDER BY id DESC LIMIT 1", (case_id,)
        ).fetchone()
        prev_hash = last["hash"] if last else None
        ts = _now()
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)[:60000]
        h = hashlib.sha256(f"{prev_hash}|{ts}|{tool}|{payload_json}".encode("utf-8")).hexdigest()
        self.conn.execute(
            "INSERT INTO evidence(case_id, ts, tool, payload_json, prev_hash, hash) VALUES(?,?,?,?,?,?)",
            (case_id, ts, tool, payload_json, prev_hash, h),
        )
        self.conn.commit()
        return {"ts": ts, "tool": tool, "hash": h}

    def verify_chain(self, case_id: int) -> dict:
        rows = self.conn.execute(
            "SELECT * FROM evidence WHERE case_id=? ORDER BY id", (case_id,)
        ).fetchall()
        prev = None
        for r in rows:
            expect = hashlib.sha256(
                f"{prev}|{r['ts']}|{r['tool']}|{r['payload_json']}".encode("utf-8")
            ).hexdigest()
            if expect != r["hash"] or (prev and r["prev_hash"] != prev):
                return {"intact": False, "broken_at_id": int(r["id"]), "n": len(rows)}
            prev = r["hash"]
        return {"intact": True, "n_evidence": len(rows)}

    # ---------------- hypothesis( + prior) ----------------
    def record_hypothesis(self, case_id: int, statement: str, verdict: str,
                          posterior: float, supersede_same_model: bool = True) -> None:
        ts = _now()
        if supersede_same_model:
            # modelmoreinhypothesis(disable)
            self.conn.execute(
                """UPDATE hypotheses SET valid_to=?
                   WHERE statement=? AND valid_to IS NULL AND case_id IN
                     (SELECT id FROM cases WHERE model_name=
                       (SELECT model_name FROM cases WHERE id=?))
                   AND case_id != ?""",
                (ts, statement, case_id, case_id),
            )
        self.conn.execute(
            "INSERT INTO hypotheses(case_id, statement, verdict, posterior, valid_from, valid_to)"
            " VALUES(?,?,?,?,?,NULL)",
            (case_id, statement, verdict, round(float(posterior), 4), ts),
        )
        self.conn.commit()

    def hypothesis_priors(self, model_name: str) -> list[dict]:
        """modelhistoryhypothesis——belowinvestigationprior."""
        rows = self.conn.execute(
            """SELECT h.*, c.model_version, c.model_name FROM hypotheses h
               JOIN cases c ON c.id = h.case_id
               WHERE c.model_name=? ORDER BY h.valid_from DESC""",
            (model_name,),
        ).fetchall()
        return [
            {
                "statement": r["statement"],
                "verdict": r["verdict"],
                "posterior": r["posterior"],
                "model_version": r["model_version"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "still_valid": r["valid_to"] is None,
            }
            for r in rows
        ]

    # ---------------- ----------------
    def record_correction(self, case_id: int, edge: str, operation: str,
                          note: str, verification: dict | None) -> None:
        self.conn.execute(
            "INSERT INTO corrections(case_id, edge, operation, note, verification_json, ts)"
            " VALUES(?,?,?,?,?,?)",
            (case_id, edge, operation, note or "",
             json.dumps(verification, ensure_ascii=False, default=str) if verification else None, _now()),
        )
        self.conn.commit()

    # ---------------- ----------------
    def history(self, model_name: str | None = None) -> list[dict]:
        q = """SELECT c.*, COUNT(DISTINCT e.id) n_evidence,
                      (SELECT COUNT(*) FROM hypotheses h WHERE h.case_id=c.id) n_hypotheses,
                      (SELECT COUNT(*) FROM corrections r WHERE r.case_id=c.id) n_corrections
               FROM cases c LEFT JOIN evidence e ON e.case_id=c.id"""
        args = ()
        if model_name:
            q += " WHERE c.model_name=?"
            args = (model_name,)
        q += " GROUP BY c.id ORDER BY c.id DESC"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def case_summary(self, case_id: int) -> dict:
        case = dict(self.conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone())
        case["chain"] = self.verify_chain(case_id)
        case["hypotheses"] = [
            dict(r) for r in self.conn.execute(
                "SELECT statement, verdict, posterior, valid_from, valid_to"
                " FROM hypotheses WHERE case_id=? ORDER BY id", (case_id,)
            ).fetchall()
        ]
        case["corrections"] = [
            dict(r) for r in self.conn.execute(
                "SELECT edge, operation, note, ts FROM corrections WHERE case_id=? ORDER BY id",
                (case_id,)
            ).fetchall()
        ]
        return case
