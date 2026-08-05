"""One-time conversion of the HisarMod2019.1 CSV distribution to .npy.

The repo's main.py scripts expect train.mat/test.mat (the authors converted the
CSVs to MATLAB files themselves and never shipped the converter). The public
HisarMod distribution is CSV, so this rebuilds the equivalent arrays directly.

Signal CSV: one row per example, 1024 comma-separated MATLAB complex literals
("6.9697+15.344i"). Emitted as float32 (N, 2, 1024) = [real, imag], matching the
(None, 2, 1024, 1) input the models expect once a trailing axis is added.

Labels: the shipped train_labels.csv uses a sparse 2-digit family code
(0,1,2,3,4,10,...,61) -- 26 distinct values, exactly 20000 examples each, NOT
0..25. to_categorical() on the raw codes would produce 62 columns against a
26-output model. They are remapped to 0..25 by ascending code, and the mapping is
written to label_code_map.json.

  NOTE: that remap is a bijection, so accuracy/loss are unaffected by it. But the
  ORDER of the resulting indices against the `classes` name list in main.py is
  UNVERIFIED -- the HisarMod docs would be needed to confirm which code is BPSK
  etc. Overall and per-SNR accuracy are correct regardless; only the class NAMES
  on a confusion matrix may be permuted.
"""
import os, sys, json, time
import numpy as np

BASE = "/Users/jake/Documents/work/amr/AMR-Benchmark/Datasets/HisarMod2019.1"
OUT = os.path.join(BASE, "npy")
CHUNK = 5000
SIG_LEN = 1024


def convert_split(split, data_csv, labels_csv, snr_csv):
    labels_raw = np.loadtxt(os.path.join(BASE, labels_csv), dtype=int)
    snr = np.loadtxt(os.path.join(BASE, snr_csv), dtype=int)
    n = len(labels_raw)
    print(f"[{split}] {n:,} examples")

    codes = np.unique(labels_raw)
    mapping = {int(c): i for i, c in enumerate(codes)}
    labels = np.array([mapping[int(c)] for c in labels_raw], dtype=np.int16)

    np.save(os.path.join(OUT, f"{split}_labels.npy"), labels)
    np.save(os.path.join(OUT, f"{split}_snr.npy"), snr.astype(np.int16))

    out_path = os.path.join(OUT, f"{split}_data.npy")
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(n, 2, SIG_LEN))
    t0 = time.time()
    row = 0
    buf = []
    with open(os.path.join(BASE, data_csv)) as fh:
        for line in fh:
            buf.append(line)
            if len(buf) == CHUNK:
                row = _flush(buf, arr, row)
                buf = []
                el = time.time() - t0
                print(f"[{split}] {row:,}/{n:,}  {row/el:,.0f} rows/s  "
                      f"eta {(n-row)/max(row/el,1)/60:.1f} min", flush=True)
    if buf:
        row = _flush(buf, arr, row)
    arr.flush()
    assert row == n, f"row count mismatch: parsed {row}, expected {n}"
    print(f"[{split}] done in {(time.time()-t0)/60:.1f} min -> {out_path}")
    return mapping


def _flush(buf, arr, row):
    block = np.array([np.array(l.strip().replace("i", "j").split(","), dtype=np.complex64)
                      for l in buf])
    arr[row:row + len(block), 0, :] = block.real
    arr[row:row + len(block), 1, :] = block.imag
    return row + len(block)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m_train = convert_split("train", "Train/train_data.csv",
                            "Train/train_labels.csv", "Train/train_snr.csv")
    m_test = convert_split("test", "Test/test_data.csv",
                           "Test/test_labels.csv", "Test/test_snr.csv")
    assert m_train == m_test, "train/test label codes differ"
    with open(os.path.join(OUT, "label_code_map.json"), "w") as fh:
        json.dump({"raw_code_to_index": m_train,
                   "note": "index order vs the `classes` name list is UNVERIFIED"},
                  fh, indent=2)
    print("\nwrote", OUT)
