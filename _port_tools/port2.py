"""Second port pass: fixes that only surface once a model is actually constructed.

Run with --apply to write changes; default is a dry-run report.
"""
import os, re, sys, glob
from collections import Counter

ROOT = "/Users/jake/Documents/work/amr/AMR-Benchmark"
DATASET_DIRS = ["HisarMod", "RML2018", "RML201610a", "RML201610b"]
APPLY = "--apply" in sys.argv
stats = Counter()


def fix_init_kwarg(src):
    """`init=` is the Keras 1.x spelling. Keras 2.2.4 still accepted it through its
    legacy-interface shim; 2.15 raises TypeError. The 2.x name is kernel_initializer."""
    out, n = re.subn(r"\binit\s*=\s*(['\"])", r"kernel_initializer=\1", src)
    stats["init= -> kernel_initializer="] += n
    return out


def fix_duplicate_reshape_name(src):
    """MCLDNN names one Reshape 'reshap2' and another 'reshape', but leaves the
    first unnamed. TF2 Keras auto-names that one exactly 'reshape', colliding with
    the explicit one. Name it 'reshap1' to match the author's existing scheme."""
    out, n = re.subn(
        r"(x2_reshape\s*=\s*Reshape\(\[-1,\s*\d+,\s*50\])\)",
        r"\1,name='reshap1')",
        src,
    )
    stats["named the unnamed MCLDNN Reshape 'reshap1'"] += n
    return out


TRANSFORMS = [fix_init_kwarg, fix_duplicate_reshape_name]

changed = []
for d in DATASET_DIRS:
    for path in glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
        src = original
        for fn in TRANSFORMS:
            src = fn(src)
        if src != original:
            changed.append(os.path.relpath(path, ROOT))
            if APPLY:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src)

print(f"{'APPLIED' if APPLY else 'DRY RUN'} — {len(changed)} files changed\n")
for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
    if v:
        print(f"  {v:5}  {k}")
