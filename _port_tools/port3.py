"""Third port pass: Keras renamed the accuracy history keys.

Keras 2.2.4 recorded 'acc'/'val_acc'; TF2 Keras records 'accuracy'/'val_accuracy'.
mltools.show_history() indexes the old names and dies with KeyError after training
completes -- i.e. only after you've already spent the epochs. Rewritten to accept
either, so the files still work against an old Keras too.

Run with --apply to write changes; default is a dry-run report.
"""
import os, re, sys, glob
from collections import Counter

ROOT = "/Users/jake/Documents/work/amr/AMR-Benchmark"
DATASET_DIRS = ["HisarMod", "RML2018", "RML201610a", "RML201610b"]
APPLY = "--apply" in sys.argv
stats = Counter()

SUBS = [
    (r"history\.history\['acc'\]",
     "history.history.get('acc', history.history.get('accuracy'))"),
    (r"history\.history\['val_acc'\]",
     "history.history.get('val_acc', history.history.get('val_accuracy'))"),
]

changed = []
for d in DATASET_DIRS:
    for path in glob.glob(os.path.join(ROOT, d, "**", "mltools.py"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
        src = original
        for pat, rep in SUBS:
            src, n = re.subn(pat, rep, src)
            stats[pat] += n
        if src != original:
            changed.append(os.path.relpath(path, ROOT))
            if APPLY:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src)

print(f"{'APPLIED' if APPLY else 'DRY RUN'} — {len(changed)} files changed\n")
for k, v in stats.items():
    if v:
        print(f"  {v:5}  {k}")
