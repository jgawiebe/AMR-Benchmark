"""Evaluate a trained CNN1 checkpoint on the HisarMod test set.

Reuses main.py's own data-loading section verbatim (everything above the
training block) so the test split, SNR filter and preprocessing cannot drift
from what the checkpoint was trained on. Respects the same AMR_MIN_SNR env var.

  ../../.venv/bin/python evaluate.py [path/to/weights.h5]
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

# main.py carries a UTF-8 BOM, hence utf-8-sig.
src = open('main.py', encoding='utf-8-sig').read().split('# perform training')[0]
ns = {'__file__': os.path.join(HERE, 'main.py'), '__name__': '__main__'}
exec(compile(src, 'main.py', 'exec'), ns)

X_test, Y_test, Z_test = ns['X_test'], ns['Y_test'], ns['Z_test']
classes = ns['classes']

import mltools
import rmlmodels.CNN2Model as cnn2

filepath = sys.argv[1] if len(sys.argv) > 1 else 'weights/CNN2_0.5.wts.h5'
if not os.path.exists(filepath):
    sys.exit(f"no checkpoint at {filepath}")

model = cnn2.CNN2Model()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.load_weights(filepath)
print(f"loaded {filepath}")
print(f"test set: {X_test.shape[0]:,} examples, SNR >= {ns['MIN_SNR']} dB\n")

batch_size = int(os.environ.get('AMR_BATCH', 400))
loss, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print(f"\ntest loss {loss:.4f}   test accuracy {acc*100:.2f}%\n")

Y_hat = model.predict(X_test, batch_size=batch_size, verbose=1)

cm, right, wrong = mltools.calculate_confusion_matrix(Y_test, Y_hat, classes)
print(f"\nOverall Accuracy: {100.0*right/(right+wrong):.2f}%  ({right} + {wrong})\n")

# per-SNR accuracy, computed here rather than via mltools so the numbers land in
# stdout as a table (mltools' version is oriented at writing figures)
print("per-SNR accuracy")
true = Y_test.argmax(1)
pred = Y_hat.argmax(1)
z = Z_test.ravel()
for lv in np.unique(z):
    m = z == lv
    print(f"  {lv:+3d} dB   {100.0*(true[m]==pred[m]).mean():6.2f}%   n={m.sum():,}")

print("\nper-class accuracy (index order; names UNVERIFIED, see PORT_HANDOFF)")
for i in range(len(classes)):
    m = true == i
    if m.sum():
        print(f"  [{i:2d}] {classes[i]:<12} {100.0*(pred[m]==i).mean():6.2f}%   n={m.sum():,}")

os.makedirs('figure', exist_ok=True)
mltools.plot_confusion_matrix(cm, labels=classes,
                              save_filename='figure/eval_confusion_matrix.png')
print("\nwrote figure/eval_confusion_matrix.png")
