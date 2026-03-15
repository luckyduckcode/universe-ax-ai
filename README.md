# Ax Universe Sim

Ax Universe Sim is a Hyperdimensional Computing (HDC) simulation with a Tkinter GUI.
It models collective cognition, Earth-channel interaction, dynamic concept emergence,
and persistent archival of discovered concept-laws.

## Core Model

The system is built around these assumptions:

1. The universe should remain **dynamic and harmonized**, not static.
2. **Earth** appears as a semantic bridge for observer interaction.
3. Prompts are encoded into HDC vectors and propagated through agent memory.
4. Population resonance determines communication quality.
5. Responses feed back into the system and reshape collective state.
6. New stable concepts can be documented as **Synthetic Laws**.

## Major Features

- **Earth Channel loop**: idea transmission, reception scoring, observer response, cycle logging.
- **Agent memory**: short conversation memory + relevance recall in responses.
- **Civilization persistence mode**: damped entropy and Earth linger windows.
- **Living Concepts**: dynamic concept birth from autonomous evolution and user interventions.
- **Law Library (persistent)**:
	- `synthetic_laws.jsonl` stores law metadata + vector snapshots.
	- `synthetic_laws_map.json` stores 2D projection (PCA or t-SNE when available).
	- Semantic similarity search over archived law vectors.
	- Reseed support to inject archived laws into future runs.
- **Scientific Method curriculum**:
	- GUI button: **🔬 Teach Scientific Method**
	- Seeds concepts such as `scientific method`, `hypothesis`, `experiment`, `analysis`,
		`reproducibility`, and `falsifiability`.

## Repository Layout

- `ax_universe_sim.py` — core simulation engine and persistence logic.
- `ax_universe_gui.py` — Tkinter GUI (Earth Channel, Plot, Concepts, Law Library).
- `law_archaeology.py` — post-run archaeology script for law extraction and influence decoding.

## Quick Start (macOS)

Use the Python 3.12 environment for GUI stability:

```bash
cd "/Users/charlie/Desktop/universe ai"
/opt/homebrew/bin/python3.12 -m venv .venv312
.venv312/bin/pip install numpy matplotlib
.venv312/bin/python ax_universe_gui.py
```

## Using the App

1. Start the GUI.
2. Let seeding complete.
3. Use toolbar actions:
	 - **✦ Divine Light** for concept acceleration.
	 - **🔬 Teach Scientific Method** for scientific-reasoning instruction.
4. Open **📜 Law Library** tab to inspect documented laws and map.
5. Use **🌱 Reseed Universe** to inject archived laws into the current run.

## Law Archaeology Script

Run:

```bash
.venv312/bin/python law_archaeology.py
```

Output:

- Extracts documented laws from JSONL/log signatures.
- Decodes likely Hebrew-root influences via cosine similarity.
- Prints an archaeological report for each discovered law.

## Notes

- Generated files `synthetic_laws.jsonl` and `synthetic_laws_map.json` are runtime artifacts.
- If t-SNE is unavailable, map projection automatically falls back to PCA/SVD.
