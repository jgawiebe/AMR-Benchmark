"""Build + train-one-step + predict every model, in a subprocess each.
Separates real port breakage from the NCHW-on-CPU hardware limitation."""
import os, sys, glob, json, subprocess, textwrap
from collections import Counter

ROOT = "/Users/jake/Documents/work/amr/AMR-Benchmark"
TMP = "/Users/jake/.claude/jobs/1a583119/tmp"

CHILD = textwrap.dedent('''
    import os, sys, warnings, importlib.util, inspect
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    path, moddir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, moddir); sys.path.insert(0, os.path.dirname(moddir))
    import numpy as np
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    VAR = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    fn = None
    for n, f in vars(m).items():
        if inspect.isfunction(f) and f.__module__ == name and not n.startswith("_"):
            sig = inspect.signature(f)
            if not [p for p in sig.parameters.values()
                    if p.kind not in VAR and p.default is inspect.Parameter.empty]:
                fn = f; break
    if fn is None:
        print("NOFACTORY"); sys.exit(0)
    model = fn()
    N = 8
    in_shapes = [t.shape for t in model.inputs]
    out_shapes = [t.shape for t in model.outputs]
    X = [np.random.randn(N, *[d or 1 for d in s[1:]]).astype("float32") for s in in_shapes]
    Y, losses = [], []
    for s in out_shapes:
        dims = [d or 1 for d in s[1:]]
        if len(dims) == 1 and dims[0] <= 32:
            Y.append(np.eye(dims[0])[np.random.randint(0, dims[0], N)].astype("float32"))
            losses.append("categorical_crossentropy")
        else:
            Y.append(np.random.randn(N, *dims).astype("float32"))
            losses.append("mse")
    model.compile(optimizer="adam", loss=losses)
    h = model.fit(X if len(X)>1 else X[0], Y if len(Y)>1 else Y[0],
                  epochs=1, batch_size=4, verbose=0)
    model.predict(X if len(X)>1 else X[0], verbose=0)
    print("TRAIN_OK", model.count_params(), [tuple(int(d) if d else -1 for d in s[1:]) for s in in_shapes])
''')

child = os.path.join(TMP, "_child_train.py")
with open(child, "w") as fh:
    fh.write(CHILD)

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*", "*", "rmlmodels", "*.py"))):
    moddir = os.path.dirname(f)
    rel = os.path.relpath(f, ROOT)
    model_dir = os.path.dirname(moddir)
    p = subprocess.run([os.path.join(ROOT, ".venv/bin/python"), child, f, moddir],
                       capture_output=True, text=True, cwd=model_dir, timeout=900)
    out, err = p.stdout.strip(), p.stderr
    if "TRAIN_OK" in out:
        status, detail = "TRAIN_OK", out.split("TRAIN_OK", 1)[1].strip()
    elif "NCHW" in err:
        status, detail = "NCHW_CPU", "channels_first Conv2D — needs CUDA GPU"
    else:
        lines = [l for l in err.splitlines() if l.strip()]
        status, detail = "FAIL", (lines[-1] if lines else "unknown")[:150]
    rows.append({"model": os.path.relpath(model_dir, ROOT), "file": rel,
                 "status": status, "detail": detail})
    print(f"{status:9} {rows[-1]['model']:26} {detail[:90]}")

with open(os.path.join(TMP, "train_all.json"), "w") as fh:
    json.dump(rows, fh, indent=2)

print("\n=== SUMMARY ===")
for k, v in Counter(r["status"] for r in rows).most_common():
    print(f"{k}: {v}")
print("\nNCHW_CPU models:")
for r in rows:
    if r["status"] == "NCHW_CPU":
        print("  ", r["model"])
print("\nFAIL models:")
for r in rows:
    if r["status"] == "FAIL":
        print("  ", r["model"], "->", r["detail"][:110])
