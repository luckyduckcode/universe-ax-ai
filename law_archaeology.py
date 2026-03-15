import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ax_universe_sim import AxUniverseSim


def extract_laws(archive_path):
    laws_found = []
    path = Path(archive_path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            # Check for JSON records first
            if raw.startswith("{"):
                obj = json.loads(raw)
                if "name" in obj or "law_name" in obj:
                    laws_found.append(obj)
            # Fallback to raw log signatures
            elif "'born' from" in raw or "NEW CONCEPT" in raw:
                laws_found.append({"raw_log": raw})

    return laws_found


def decode_law_origin(
    law_vector: np.ndarray,
    lexicon_vectors: Dict[str, np.ndarray],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Return top Hebrew root influences by cosine similarity."""
    if law_vector.ndim != 1:
        law_vector = law_vector.reshape(-1)
    lv_norm = float(np.linalg.norm(law_vector))
    if lv_norm == 0.0:
        return []

    scores: List[Tuple[str, float]] = []
    for root, vec in lexicon_vectors.items():
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        vv_norm = float(np.linalg.norm(vec))
        if vv_norm == 0.0:
            continue
        sim = float(np.dot(law_vector, vec) / (lv_norm * vv_norm))
        scores.append((root, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_lexicon_vectors(sim: AxUniverseSim) -> Dict[str, np.ndarray]:
    """Encode Hebrew roots into HDC vectors for probing."""
    if not getattr(sim, "hebrew_lexicon", None):
        sim.load_hebrew_dictionary()
    vectors: Dict[str, np.ndarray] = {}
    for _, entry in sim.hebrew_lexicon.items():
        root = str(entry.get("root", "")).strip()
        if not root:
            continue
        tr = str(entry.get("tr", "")).strip()
        primary = str(entry.get("primary", "")).strip()
        probe_text = " ".join(part for part in [root, tr, primary] if part)
        vec = sim.encode_text_to_hdc(probe_text).astype(np.float32)
        vectors[root] = vec
    return vectors


def report_laws(archive_path: str) -> None:
    laws = extract_laws(archive_path)
    print("--- Archaeological Report: Synthetic Laws ---")
    if not laws:
        print("No documented laws found.")
        return

    sim = AxUniverseSim(num_agents=8)
    lexicon_vectors = build_lexicon_vectors(sim)

    for law in laws:
        if "raw_log" in law:
            print(f"Log Artifact: {law['raw_log']}")
            continue

        name = law.get("name") or law.get("law_name") or "unknown_law"
        snap = law.get("vector_snapshot")
        if isinstance(snap, list) and len(snap) > 0:
            law_vector = np.array(snap, dtype=np.float32)
            influences = decode_law_origin(law_vector, lexicon_vectors, top_k=3)
        else:
            influences = []

        print(f"Law: '{name}'")
        if influences:
            print(f"Influences: {[(r, round(s, 3)) for r, s in influences]}")
        else:
            print("Influences: []")


if __name__ == "__main__":
    default_archive = Path(__file__).with_name("synthetic_laws.jsonl")
    report_laws(str(default_archive))
