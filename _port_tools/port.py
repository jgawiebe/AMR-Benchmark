"""Port AMR-Benchmark from Keras 2.2.4 / TF 1.14 to Keras 2.15 / TF 2.15 (arm64).

Run with --apply to write changes; default is a dry-run report.
"""
import os, re, sys, glob
from collections import Counter

ROOT = "/Users/jake/Documents/work/amr/AMR-Benchmark"
DATASET_DIRS = ["HisarMod", "RML2018", "RML201610a", "RML201610b"]
APPLY = "--apply" in sys.argv

stats = Counter()


def fix_submodule_imports(src):
    """Keras 2.13+ removed these import shims; the names live on the parent package."""
    out, n = re.subn(r"from keras\.layers\.(?:convolutional|core) import", "from keras.layers import", src)
    stats["keras.layers.{convolutional,core} -> keras.layers"] += n
    out, n = re.subn(r"from keras\.utils\.(?:vis_utils|np_utils) import", "from keras.utils import", out)
    stats["keras.utils.{vis_utils,np_utils} -> keras.utils"] += n
    return out


def fix_cudnn(src):
    """CuDNNLSTM/CuDNNGRU are gone. In TF2 Keras, LSTM/GRU default to the same
    math (tanh + sigmoid, GRU reset_after=True) and pick a cuDNN kernel when
    one is available, so the plain layers are the faithful replacement.
    \\b keeps `rmlmodels.CuDNNLSTMModel` (a module name) untouched."""
    out, n = re.subn(r"\bCuDNNLSTM\b", "LSTM", src)
    stats["CuDNNLSTM -> LSTM"] += n
    out, n2 = re.subn(r"\bCuDNNGRU\b", "GRU", out)
    stats["CuDNNGRU -> GRU"] += n2
    if n or n2:
        out = dedupe_layer_imports(out)
    return out


def dedupe_layer_imports(src):
    """The rename can produce `from keras.layers import LSTM,LSTM,Flatten`."""
    def repl(m):
        head, names = m.group(1), m.group(2)
        seen, kept = set(), []
        for raw in names.split(","):
            name = raw.strip()
            if name and name not in seen:
                seen.add(name)
                kept.append(name)
        return head + ",".join(kept)

    out, n = re.subn(r"^(\s*from keras\.layers import\s+)(.+)$", repl, src, flags=re.M)
    return out


def fix_optimizer(src):
    """`decay` is a hard ValueError in the Keras 2.15 optimizer; `lr` is deprecated.
    `epsilon=None` meant 'backend default' (1e-7) in 2.2.4, so spell it out."""
    def repl(m):
        args = m.group(2)
        args = re.sub(r"\blr\s*=", "learning_rate=", args)
        args = re.sub(r",?\s*\bdecay\s*=\s*[0-9.eE+-]+", "", args)
        args = re.sub(r"\bepsilon\s*=\s*None\b", "epsilon=1e-07", args)
        return m.group(1) + "(" + args + ")"

    out, n = re.subn(r"\b(Adam|SGD|RMSprop|Adagrad|Adadelta|Nadam|Adamax)\(([^()]*)\)", repl, src)
    changed = sum(1 for _ in re.finditer(r"learning_rate=", out)) - sum(1 for _ in re.finditer(r"learning_rate=", src))
    stats["optimizer lr=/decay=/epsilon=None fixed"] += changed
    return out


# NOTE: no nb_epoch transform. The repo already calls fit(epochs=nb_epoch, ...);
# `nb_epoch` is only a local variable name, so rewriting it would leave the
# `epochs=nb_epoch` call sites referencing an undefined name.


def fix_dead_imports(src):
    """keras.backend.tensorflow_backend is gone in TF2; multi_gpu_model was removed
    (and is imported but never called here)."""
    lines = src.splitlines(keepends=True)
    kept = []
    for line in lines:
        if re.match(r"\s*import keras\.backend\.tensorflow_backend\b", line):
            stats["dropped `import keras.backend.tensorflow_backend`"] += 1
            continue
        if re.match(r"\s*from keras\.utils import multi_gpu_model\s*$", line):
            stats["dropped `multi_gpu_model` import (unused)"] += 1
            continue
        kept.append(line)
    return "".join(kept)


def fix_tf1_session(src, relpath):
    """The one file with TF1 session plumbing. allow_growth is a CUDA concern and
    has no meaning on this machine; TF2 has no tf.ConfigProto/tf.Session."""
    if "tf.ConfigProto" not in src:
        return src
    out = re.sub(
        r"^config = tf\.ConfigProto\(\)\nconfig\.gpu_options\.allow_growth\s*=\s*True\nsess = tf\.Session\(config=config\)\n+KTF\.set_session\(sess\)\n",
        "",
        src,
        flags=re.M,
    )
    if out != src:
        stats["removed TF1 session block"] += 1
    return out


TRANSFORMS = [fix_submodule_imports, fix_cudnn, fix_optimizer, fix_dead_imports]

changed_files = []
for d in DATASET_DIRS:
    for path in glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
        src = original
        for fn in TRANSFORMS:
            src = fn(src)
        src = fix_tf1_session(src, path)
        if src != original:
            changed_files.append(os.path.relpath(path, ROOT))
            if APPLY:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src)

print(f"{'APPLIED' if APPLY else 'DRY RUN'} — {len(changed_files)} files changed\n")
for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
    if v:
        print(f"  {v:5}  {k}")
