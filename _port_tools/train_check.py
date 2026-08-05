"""Build each model, feed it correctly-shaped random data, and run one real
training step + a predict. Catches breakage that model construction alone misses."""
import os, sys, warnings, importlib.util, inspect
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = "/Users/jake/Documents/work/amr/AMR-Benchmark"
TARGETS = [
    ("RML201610a/LSTM2", "rmlmodels/CuDNNLSTMModel.py"),   # was CuDNNLSTM
    ("RML201610a/GRU2", "rmlmodels/GRUModel.py"),          # was CuDNNGRU
    ("RML201610a/CNN1", "rmlmodels/CNN2Model.py"),         # keras.layers.core
    ("RML201610a/CLDNN", "rmlmodels/CLDNNLikeModel.py"),   # init= kwarg
    ("RML201610a/MCLDNN", "rmlmodels/MCLDNN.py"),          # duplicate layer name, multi-input
    ("RML201610a/PET-CGDNN", "rmlmodels/PETCGDNN.py"),     # Lambda / custom ops
    ("RML201610a/DAE", "rmlmodels/DAE.py"),                # multi-output autoencoder
    ("RML201610a/ResNet", "rmlmodels/ResNet.py"),
]

import numpy as np

BASE_PATH = list(sys.path)  # keep site-packages; only the repo dirs rotate per model

results = []
for reldir, relfile in TARGETS:
    d = os.path.join(ROOT, reldir)
    os.chdir(d)
    sys.path[:] = [os.path.join(d, "rmlmodels"), d] + BASE_PATH
    name = os.path.splitext(os.path.basename(relfile))[0]
    try:
        for mod in [m for m in list(sys.modules) if m == name]:
            del sys.modules[mod]
        spec = importlib.util.spec_from_file_location(name, os.path.join(d, relfile))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        VAR = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        fn = None
        for n, f in vars(m).items():
            if inspect.isfunction(f) and f.__module__ == name and not n.startswith("_"):
                sig = inspect.signature(f)
                if not [p for p in sig.parameters.values()
                        if p.kind not in VAR and p.default is inspect.Parameter.empty]:
                    fn = f
                    break
        model = fn()

        N = 8
        def rand_for(shapes):
            return [np.random.randn(N, *[d or 1 for d in s[1:]]).astype("float32") for s in shapes]

        in_shapes = [t.shape for t in model.inputs]
        out_shapes = [t.shape for t in model.outputs]
        X = rand_for(in_shapes)
        Y = []
        for s in out_shapes:
            dims = [d or 1 for d in s[1:]]
            if len(dims) == 1 and dims[0] <= 32:      # classifier head -> one-hot
                y = np.eye(dims[0])[np.random.randint(0, dims[0], N)].astype("float32")
            else:
                y = np.random.randn(N, *dims).astype("float32")
            Y.append(y)

        model.compile(optimizer="adam",
                      loss=["categorical_crossentropy" if (len(s) == 2 and (s[1] or 0) <= 32)
                            else "mse" for s in out_shapes])
        h = model.fit(X if len(X) > 1 else X[0],
                      Y if len(Y) > 1 else Y[0],
                      epochs=1, batch_size=4, verbose=0)
        p = model.predict(X if len(X) > 1 else X[0], verbose=0)
        loss = list(h.history.values())[0][0]
        results.append((reldir, "OK", f"params={model.count_params():,} loss={loss:.4f} "
                                      f"in={[tuple(s[1:]) for s in in_shapes]}"))
    except Exception as e:
        results.append((reldir, "FAIL", f"{type(e).__name__}: {str(e)[:160]}"))

print()
for r, status, detail in results:
    print(f"{status:5} {r:26} {detail}")
print(f"\n{sum(1 for _,s,_ in results if s=='OK')}/{len(results)} trained + predicted")
