"""Synthetic IQ signal generator matching HisarMod2019.1's conventions.

Produces (N, 2, 1024, 1) float32 batches ready to feed the trained CNN1.

The conventions here were measured from the training data, not assumed:

POWER (the critical one).  HisarMod does NOT normalise examples. Signal power is
fixed at exactly 1.0 and AWGN is added on top with power 10^(-SNR/10), with no
rescaling afterwards. Measured total power tracks 1 + 10^(-SNR/10) to within 0.3%
across all 20 SNR levels. Total power is therefore a monotonic function of SNR
and the network can read SNR straight off the amplitude -- so normalising the
final noisy signal to unit power (the usual instinct) puts every example off
distribution. add_awgn() below is the only place noise is added, and it
deliberately does not renormalise.

LAYOUT.  Channel 0 = real, channel 1 = imaginary. Not amplitude/phase -- other
models in this repo use that convention, CNN1 does not. 1024 samples is fixed by
the Flatten layer.

SYMBOL RATE.  HisarMod oversamples very little. Its 99% occupied bandwidth is
0.50-0.56 of fs (measured per class at +18 dB), which for RRC with beta=0.35
implies roughly 2-3 samples/symbol. SPS_RANGE is set accordingly.

  Do not trust a cyclostationary estimate averaged across classes here -- that
  suggests ~22 sps, which is an artefact of averaging and produces signals ~45%
  too narrow. Probing the model with 6-32 sps collapses 22 of 26 modulations onto
  a single output index; with sps=2-3 they spread over 13-14 indices. Bandwidth
  match matters more than anything else in this file.

  Caveat: at sps=2 the generated bandwidth is 0.61 vs the 0.50-0.56 target, at
  sps=3 it is 0.43. Neither matches exactly, and integer upsampling cannot hit
  the ~2.4 sps the target implies. Some mismatch remains.

CAVEAT.  These are textbook constructions, not a reproduction of HisarMod's
generator. Notably HisarMod applies fading channels (ideal/static/Rayleigh/
Rician/Nakagami per its paper) which are only crudely approximated by
`fading=` here. Divergence between this generator and HisarMod is the point --
it is what makes these signals an independent test of the model.
"""
import numpy as np

N_SAMPLES = 1024
SPS_RANGE = (2, 5)           # samples/symbol drawn per example unless pinned
DEFAULT_BETA = 0.35          # RRC rolloff

# HisarMod's 26 classes, in the order main.py lists them.
CLASSES = [
    'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '64PSK',
    '4QAM', '8QAM', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
    '2FSK', '4FSK', '8FSK', '16FSK',
    '4PAM', '8PAM', '16PAM',
    'AM-DSB', 'AM-DSB-SC', 'AM-USB', 'AM-LSB', 'FM', 'PM',
]

DIGITAL = set(CLASSES[:20])
ANALOG = set(CLASSES[20:])


# --------------------------------------------------------------------------
# constellations
# --------------------------------------------------------------------------

def psk_constellation(M):
    return np.exp(2j * np.pi * np.arange(M) / M)


def _square_qam(M):
    side = int(round(np.sqrt(M)))
    assert side * side == M
    lv = 2 * np.arange(side) - side + 1
    I, Q = np.meshgrid(lv, lv)
    return (I + 1j * Q).ravel()


def _cross_qam(M):
    """Standard cross-QAM for the non-square orders HisarMod carries."""
    if M == 8:                                    # 2x4 rectangular
        I = np.array([-3, -1, 1, 3])
        Q = np.array([-1, 1])
        Ig, Qg = np.meshgrid(I, Q)
        return (Ig + 1j * Qg).ravel()
    if M == 32:                                   # 6x6 minus 4 corners
        lv = np.array([-5, -3, -1, 1, 3, 5])
        I, Q = np.meshgrid(lv, lv)
        pts = (I + 1j * Q).ravel()
        return np.array([p for p in pts if not (abs(p.real) == 5 and abs(p.imag) == 5)])
    if M == 128:                                  # 12x12 minus 2x2 corner blocks
        lv = np.arange(-11, 12, 2)
        I, Q = np.meshgrid(lv, lv)
        pts = (I + 1j * Q).ravel()
        return np.array([p for p in pts
                         if not (abs(p.real) >= 9 and abs(p.imag) >= 9)])
    raise ValueError(f"no cross-QAM construction for M={M}")


def qam_constellation(M):
    c = _square_qam(M) if int(round(np.sqrt(M))) ** 2 == M else _cross_qam(M)
    return c / np.sqrt(np.mean(np.abs(c) ** 2))   # unit average power


def pam_constellation(M):
    c = (2 * np.arange(M) - M + 1).astype(complex)
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


# --------------------------------------------------------------------------
# pulse shaping
# --------------------------------------------------------------------------

def rrc_filter(beta, span, sps):
    """Root-raised-cosine, unit energy."""
    N = int(span * sps)
    t = (np.arange(N + 1) - N / 2.0) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif beta > 0 and abs(abs(ti) - 1.0 / (4 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def _shape(symbols, sps, beta):
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = rrc_filter(beta, span=8, sps=sps)
    return np.convolve(up, h, mode='same')


# --------------------------------------------------------------------------
# message signal for the analog modulations
# --------------------------------------------------------------------------

def _message(n, rng, bw=0.05):
    """Band-limited real message, peak-normalised."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(n) > bw] = 0
    m = np.fft.irfft(X, n)
    peak = np.max(np.abs(m))
    return m / peak if peak > 0 else m


def _hilbert(x):
    """Analytic-signal imaginary part, via FFT (avoids a scipy dependency)."""
    n = len(x)
    X = np.fft.fft(x)
    hsel = np.zeros(n)
    hsel[0] = 1
    if n % 2 == 0:
        hsel[n // 2] = 1
        hsel[1:n // 2] = 2
    else:
        hsel[1:(n + 1) // 2] = 2
    return np.imag(np.fft.ifft(X * hsel))


# --------------------------------------------------------------------------
# per-modulation baseband synthesis
# --------------------------------------------------------------------------

def _digital_baseband(mod, n_out, sps, beta, rng):
    n_sym = int(np.ceil(n_out / sps)) + 16

    if mod.endswith('PSK'):
        M = 2 if mod == 'BPSK' else (4 if mod == 'QPSK' else int(mod[:-3]))
        const = psk_constellation(M)
        sym = const[rng.integers(0, M, n_sym)]
        return _shape(sym, sps, beta)

    if mod.endswith('QAM'):
        M = int(mod[:-3])
        const = qam_constellation(M)
        sym = const[rng.integers(0, M, n_sym)]
        return _shape(sym, sps, beta)

    if mod.endswith('PAM'):
        M = int(mod[:-3])
        const = pam_constellation(M)
        sym = const[rng.integers(0, M, n_sym)]
        return _shape(sym, sps, beta)      # real-valued; carrier phase rotates it

    if mod.endswith('FSK'):
        M = int(mod[:-3])
        # Tones spread symmetrically inside +-MAX_DEV so high-order FSK still fits
        # under Nyquist; a fixed modulation index would alias for 16FSK at low sps.
        MAX_DEV = 0.35
        tones = np.linspace(-MAX_DEV, MAX_DEV, M)
        f = np.repeat(tones[rng.integers(0, M, n_sym)], sps)
        return np.exp(2j * np.pi * np.cumsum(f))   # continuous phase

    raise ValueError(f"unknown digital modulation {mod}")


def _analog_baseband(mod, n_out, rng):
    n = n_out + 256
    m = _message(n, rng)
    if mod == 'AM-DSB':
        return (1.0 + 0.8 * m).astype(complex)      # carrier present
    if mod == 'AM-DSB-SC':
        return m.astype(complex)                    # carrier suppressed
    if mod == 'AM-USB':
        return m + 1j * _hilbert(m)
    if mod == 'AM-LSB':
        return m - 1j * _hilbert(m)
    if mod == 'FM':
        return np.exp(2j * np.pi * 0.05 * np.cumsum(m))
    if mod == 'PM':
        return np.exp(1j * 1.5 * m)
    raise ValueError(f"unknown analog modulation {mod}")


# --------------------------------------------------------------------------
# channel + noise
# --------------------------------------------------------------------------

def _apply_fading(x, kind, rng):
    """Crude multipath. HisarMod's real channel models are richer than this."""
    if kind in (None, 'ideal'):
        return x
    if kind == 'static':
        taps = np.array([1.0, 0.3, 0.1], dtype=complex)
    elif kind == 'rayleigh':
        n = 3
        taps = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2 * n)
    elif kind == 'rician':
        n = 3
        K = 4.0
        d = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2 * n)
        taps = d.copy()
        taps[0] += np.sqrt(K)
    else:
        raise ValueError(f"unknown fading model {kind}")
    taps /= np.sqrt(np.sum(np.abs(taps) ** 2))
    return np.convolve(x, taps, mode='same')


def add_awgn(sig, snr_db, rng):
    """Unit-power signal + AWGN at snr_db. Does NOT renormalise afterwards --
    see the module docstring; the absolute level is what encodes SNR."""
    p = np.mean(np.abs(sig) ** 2)
    if p <= 0:
        raise ValueError("zero-power signal")
    sig = sig / np.sqrt(p)                       # signal power == 1.0 exactly
    npow = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(npow / 2.0) * (rng.standard_normal(len(sig))
                                   + 1j * rng.standard_normal(len(sig)))
    return sig + noise


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def generate(mod, snr_db, n=1, sps=None, beta=DEFAULT_BETA, cfo=None,
             fading=None, seed=None):
    """Generate `n` examples of `mod` at `snr_db`.

    mod     one of CLASSES
    sps     samples/symbol; None draws from SPS_RANGE per example
    cfo     carrier freq offset in cycles/sample; None draws +-0.001
    fading  None|'ideal'|'static'|'rayleigh'|'rician'
    returns (n, 2, 1024, 1) float32, ready for model.predict
    """
    if mod not in CLASSES:
        raise ValueError(f"unknown modulation {mod!r}")
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2, N_SAMPLES, 1), dtype=np.float32)

    for i in range(n):
        this_sps = sps if sps is not None else int(rng.integers(*SPS_RANGE))
        need = N_SAMPLES + 4 * this_sps + 64      # slack for transients + timing

        if mod in ANALOG:
            x = _analog_baseband(mod, need, rng)
        else:
            x = _digital_baseband(mod, need, this_sps, beta, rng)

        x = _apply_fading(x, fading, rng)

        # random carrier phase; also what rotates real-valued PAM off the I axis
        x = x * np.exp(1j * rng.uniform(0, 2 * np.pi))
        f_off = rng.uniform(-0.001, 0.001) if cfo is None else cfo
        x = x * np.exp(2j * np.pi * f_off * np.arange(len(x)))

        # random timing offset, and crop clear of the filter transient
        start = int(rng.integers(2 * this_sps, 2 * this_sps + 32))
        x = x[start:start + N_SAMPLES]
        if len(x) < N_SAMPLES:
            raise RuntimeError(f"short signal for {mod}: {len(x)}")

        x = add_awgn(x, snr_db, rng)
        out[i, 0, :, 0] = x.real
        out[i, 1, :, 0] = x.imag

    return out


def sanity_check(verbose=True):
    """Verify the generator reproduces HisarMod's measured power convention."""
    rng_seed = 0
    ok = True
    rows = []
    for snr in [-20, -10, 0, 8, 18]:
        x = generate('QPSK', snr, n=64, seed=rng_seed)
        iq = x[:, 0, :, 0] + 1j * x[:, 1, :, 0]
        meas = np.mean(np.abs(iq) ** 2)
        pred = 1 + 10 ** (-snr / 10)
        rows.append((snr, meas, pred, meas / pred))
        if not 0.9 < meas / pred < 1.1:
            ok = False
    if verbose:
        print("power convention (target: ratio ~1.0)")
        print(f"  {'SNR':>4} {'measured':>10} {'expected':>10} {'ratio':>7}")
        for snr, meas, pred, r in rows:
            print(f"  {snr:+4d} {meas:10.4f} {pred:10.4f} {r:7.4f}")
    return ok


if __name__ == '__main__':
    print(f"{len(CLASSES)} classes\n")
    ok = sanity_check()
    print()
    for mod in CLASSES:
        x = generate(mod, 18, n=2, seed=1)
        iq = x[:, 0, :, 0] + 1j * x[:, 1, :, 0]
        print(f"  {mod:<11} shape {x.shape}  power {np.mean(np.abs(iq)**2):.4f}")
    print("\npower convention:", "OK" if ok else "FAILED")
