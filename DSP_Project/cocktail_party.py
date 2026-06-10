import numpy as np
from sklearn.decomposition import FastICA
import scipy.io.wavfile as wav


def normalize_signal(signal):
    
    max_val = np.max(np.abs(signal))

    if max_val == 0:
        return np.array(signal, dtype=np.float32)

    return np.array([x / max_val for x in signal], dtype=np.float32)


def apply_ica(cont1, cont2):
    length = min(len(cont1), len(cont2))
    cont1 = cont1[:length]
    cont2 = cont2[:length]

    cont1 = cont1.astype(np.float32)
    cont2 = cont2.astype(np.float32)

    cont1 = normalize_signal(cont1)
    cont2 = normalize_signal(cont2)

    X = np.c_[cont1, cont2]

    ica = FastICA(n_components=2, whiten='unit-variance', random_state=42)
    S = ica.fit_transform(X)

    s1 = normalize_signal(S[:, 0])
    s2 = normalize_signal(S[:, 1])

    s1_out = (s1 * 32767).astype(np.int16)
    s2_out = (s2 * 32767).astype(np.int16)

    return s1_out, s2_out


def process_audio_file(file1, file2, out1, out2):
    try:
        fs1, cont1 = wav.read(file1)
        fs2, cont2 = wav.read(file2)
    except FileNotFoundError:
        print("Error: waves are not there.")
        return

    if fs1 != fs2:
        print("Error: Sampling rates do not match!")
        return

    if len(cont1.shape) > 1:
        cont1 = cont1.mean(axis=1)
    if len(cont2.shape) > 1:
        cont2 = cont2.mean(axis=1)

    s1_out, s2_out = apply_ica(cont1, cont2)

    wav.write(out1, fs1, s1_out)
    wav.write(out2, fs1, s2_out)

    print(f"Done! Saved '{out1}' and '{out2}'.")


if __name__ == "__main__":
    process_audio_file(
        "m2-2-1.wav",
        "m2-2-2.wav",
        "separated_1.wav",
        "separated_2.wav"
    )