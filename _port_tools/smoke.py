"""Import + build every rmlmodels/*.py to surface Keras 2.2.4 -> 2.15 breakages."""
import os, sys, glob, subprocess, json, textwrap

ROOT = "/Users/jake/Documents/work/amr/AMR-Benchmark"

CHILD = textwrap.dedent('''
    import sys, os, importlib.util, inspect, warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    path, moddir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, moddir)
    sys.path.insert(0, os.path.dirname(moddir))
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("IMPORT_OK")
    # try to build the model via a zero-arg-able factory function
    fns = [(n, f) for n, f in vars(mod).items()
           if inspect.isfunction(f) and f.__module__ == mod.__name__
           and not n.startswith("_")]
    built = False
    VAR = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    for n, f in fns:
        sig = inspect.signature(f)
        required = [p for p in sig.parameters.values()
                    if p.kind not in VAR and p.default is inspect.Parameter.empty]
        if not required:
            f()
            print("BUILD_OK", n)
            built = True
            break
    if not built:
        print("BUILD_SKIP", [n for n, _ in fns])
''')

child_path = "/Users/jake/.claude/jobs/1a583119/tmp/_child.py"
with open(child_path, "w") as fh:
    fh.write(CHILD)

files = sorted(glob.glob(os.path.join(ROOT, "*", "*", "rmlmodels", "*.py")))
results = []
for f in files:
    moddir = os.path.dirname(f)
    rel = os.path.relpath(f, ROOT)
    p = subprocess.run(
        [os.path.join(ROOT, ".venv/bin/python"), child_path, f, moddir],
        capture_output=True, text=True, cwd=os.path.dirname(moddir), timeout=300,
    )
    out, err = p.stdout.strip(), p.stderr.strip()
    if p.returncode == 0:
        status = "BUILD_OK" if "BUILD_OK" in out else ("IMPORT_OK" if "IMPORT_OK" in out else "OK?")
        detail = ""
    else:
        status = "FAIL"
        lines = [l for l in err.splitlines() if l.strip()]
        detail = lines[-1] if lines else "unknown"
    results.append({"file": rel, "status": status, "detail": detail})
    print(f"{status:10} {rel}  {detail[:150]}")

with open("/Users/jake/.claude/jobs/1a583119/tmp/smoke_results.json", "w") as fh:
    json.dump(results, fh, indent=2)

print("\n=== SUMMARY ===")
from collections import Counter
for k, v in Counter(r["status"] for r in results).items():
    print(f"{k}: {v}")
