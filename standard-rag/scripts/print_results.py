"""Print the results table from results.json. Called by `make results`."""

import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results.json")

if not results_path.exists():
    print(f"No {results_path} found. Run make eval-mistral first.")
    sys.exit(1)

r = json.loads(results_path.read_text())
print(f"\n{'Model':<35} | {'TriviaQA (ours)':>15} | {'TriviaQA (paper)':>16}")
print("-" * 72)
print(f"{r['model'].split('/')[-1]:<35} | {r['accuracy']:>14.2f}% | {r['paper_accuracy']:>15.2f}%")
print()
print(f"  n={r['total']}  correct={r['correct']}  "
      f"avg_retrieval={r.get('avg_retrieval_latency_ms', '?')} ms  "
      f"avg_generation={r.get('avg_generation_latency_ms', '?')} ms")
