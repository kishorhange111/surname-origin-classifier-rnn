# ruff: noqa: E501 - regex/format pairs read best on one line
"""Rewrite the README's result numbers from results/metrics.json after a retrain.

python scripts/sync_readme.py

Retraining on a different machine or PyTorch build can change the last digits
of the metrics. This keeps the README in step with the shipped results, so the
consistency test (tests/test_results_consistency.py) keeps passing.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    t, d = m["test"], m["data"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # Per-language table: one row per language, in label order.
    rows = "\n".join(
        f"| {k} | {d['names_per_language'][k]:,} | {v['support']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |"
        for k, v in t["per_class"].items()
    )
    header = r"(\| Language \| Names in dataset \| Test names \| Precision \| Recall \| F1 \|\n\|---\|---\|---\|---\|---\|---\|\n)(?:\|.*\|\n)+"
    readme = re.sub(header, lambda mm: mm.group(1) + rows + "\n", readme)

    # Headline numbers.
    subs = [
        (
            r"Held-out, stratified test set: \*\*[\d,]+ names\*\*",
            f"Held-out, stratified test set: **{d['test_names']:,} names**",
        ),
        (r"\| Accuracy \| \*\*[\d.]+ %\*\* \|", f"| Accuracy | **{100 * t['accuracy']:.1f} %** |"),
        (
            r"(\| Macro F1 \(every language weighted equally\) \| \*\*)[\d.]+(\*\* \|)",
            rf"\g<1>{t['macro_f1']:.3f}\g<2>",
        ),
        (r"(\| Weighted F1 \(weighted by language size\) \| )[\d.]+( \|)", rf"\g<1>{t['weighted_f1']:.3f}\g<2>"),
        (r"Arabic F1 is [\d.]+", f"Arabic F1 is {t['per_class']['Arabic']['f1']:.3f}"),
        (
            r"overall accuracy went from 78\.1 % to [\d.]+ %",
            f"overall accuracy went from 78.1 % to {100 * t['accuracy']:.1f} %",
        ),
        (r"macro F1 from 0\.408 to \d\.\d+", f"macro F1 from 0.408 to {t['macro_f1']:.3f}"),
        (r"27 epochs \(\d+ min on CPU\)", f"27 epochs ({round(m['training']['train_seconds'] / 60)} min on CPU)"),
    ]
    for pattern, repl in subs:
        readme = re.sub(pattern, repl, readme)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("README updated from results/metrics.json")


if __name__ == "__main__":
    main()
