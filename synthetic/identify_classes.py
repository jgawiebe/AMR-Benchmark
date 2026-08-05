"""Probe a trained CNN1 with synthetic signals of known modulation.

Two purposes:

1. Resolve the class-index -> class-name mapping. HisarMod ships labels as sparse
   2-digit family codes (0,1,2,3,4,10,...,61) which the loader remaps to 0..25 by
   ascending code. That remap is a bijection so accuracy is unaffected, but which
   index means BPSK has never been confirmed against the dataset docs. Feeding a
   known modulation and reading the argmax identifies it empirically.

2. Test whether the model generalises off HisarMod's generator at all. Test
   accuracy on HisarMod is 99.66%, but the dataset carries at least two shortcuts
   (absolute power encodes SNR exactly; some classes carry deterministic spectral
   structure). Independently generated signals are the honest test.

  ../../.venv/bin/python identify_classes.py [--snr 18] [--n 64] [weights.h5]

Interpretation: a coherent mapping (each generated modulation lighting up a
distinct index, high confidence) means the model learned transferable features.
Scattered or collapsed predictions mean it learned HisarMod.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CNN1 = os.path.join(ROOT, 'HisarMod', 'CNN1')
sys.path.insert(0, HERE)
sys.path.insert(0, CNN1)

import siggen

ap = argparse.ArgumentParser()
ap.add_argument('weights', nargs='?',
                default=os.path.join(CNN1, 'weights', 'CNN2_0.5.wts.h5'))
ap.add_argument('--snr', type=float, default=18.0)
ap.add_argument('--n', type=int, default=64, help='examples per modulation')
ap.add_argument('--sps', type=int, default=None, help='pin samples/symbol')
ap.add_argument('--fading', default=None)
ap.add_argument('--json', default=None, help='write mapping to this path')
args = ap.parse_args()

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import warnings
warnings.filterwarnings('ignore')
import rmlmodels.CNN2Model as cnn2

if not os.path.exists(args.weights):
    sys.exit(f"no checkpoint at {args.weights}")
model = cnn2.CNN2Model()
model.load_weights(args.weights)
print(f"loaded {args.weights}")
print(f"probing with {args.n} examples/modulation at {args.snr:+.0f} dB "
      f"(sps={'random' if args.sps is None else args.sps}, "
      f"fading={args.fading or 'none'})\n")

rows = []
for i, mod in enumerate(siggen.CLASSES):
    X = siggen.generate(mod, args.snr, n=args.n, sps=args.sps,
                        fading=args.fading, seed=1000 + i)
    P = model.predict(X, batch_size=min(args.n, 128), verbose=0)
    pred = P.argmax(1)
    top = np.bincount(pred, minlength=26).argmax()
    frac = float((pred == top).mean())
    conf = float(P[:, top].mean())
    rows.append({'generated': mod, 'top_index': int(top),
                 'agreement': frac, 'mean_confidence': conf,
                 'nominal_name_at_index': siggen.CLASSES[top]})

print(f"{'generated':<12} -> {'idx':>3}  {'agree':>6}  {'conf':>6}   nominal name at that index")
for r in rows:
    flag = '' if r['generated'] == r['nominal_name_at_index'] else '  <- differs'
    print(f"{r['generated']:<12} -> {r['top_index']:>3}  "
          f"{r['agreement']*100:5.1f}%  {r['mean_confidence']*100:5.1f}%   "
          f"{r['nominal_name_at_index']}{flag}")

idxs = [r['top_index'] for r in rows]
n_distinct = len(set(idxs))
n_match = sum(r['generated'] == r['nominal_name_at_index'] for r in rows)
mean_agree = float(np.mean([r['agreement'] for r in rows]))

print(f"\ndistinct indices hit: {n_distinct}/26")
print(f"agree with nominal name order: {n_match}/26")
print(f"mean within-modulation agreement: {mean_agree*100:.1f}%")

if n_distinct < 13:
    print("\nMost generated modulations collapse onto a few indices. The model is not\n"
          "transferring to independently generated signals -- consistent with it\n"
          "keying on HisarMod-specific structure rather than modulation itself.")
elif n_match >= 20:
    print("\nMapping largely matches the nominal name order, so the `classes` list in\n"
          "main.py can be trusted as-is.")
elif n_distinct >= 20:
    print("\nPredictions are well spread but do not follow the nominal name order --\n"
          "consistent with the label remap being a permutation of the `classes` list.\n"
          "Use the mapping above to relabel confusion matrices.")
else:
    print("\nMixed result: partial separation. Inspect per-modulation rows above before\n"
          "drawing conclusions; try --snr 18 --sps 22 to reduce generator mismatch.")

if args.json:
    with open(args.json, 'w') as fh:
        json.dump({'snr': args.snr, 'n': args.n, 'sps': args.sps,
                   'fading': args.fading, 'rows': rows}, fh, indent=2)
    print(f"\nwrote {args.json}")
