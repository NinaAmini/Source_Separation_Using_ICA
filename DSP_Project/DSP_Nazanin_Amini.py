import numpy as np
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA

def seperate_voice():
    try:
        fs1, cont1 = wav.read('m2-2-1.wav')
        fs2, cont2 = wav.read('m2-2-2.wav')
    except FileNotFoundError:
        print("Error:waves are not there.")
        return

    if fs1 != fs2:
        print("Sampling rates per second / no match .")
        return

    if len(cont1.shape) > 1:    #if (len(cont1.shape) > 1): cont1 = cont1[:, 0]
        cont1 = cont1.mean(axis=1)
    if len(cont2.shape) > 1:
        cont2 = cont2.mean(axis=1)

    length = min(len(cont1), len(cont2))
    cont1 = cont1[:length]
    cont2 = cont2[:length]

    cont1 = cont1.astype(np.float32)
    cont2 = cont2.astype(np.float32)


    abs_cont1 = [abs(x) for x in cont1]
    max_val = max(abs_cont1)
    cont1 = [x / max_val for x in cont1]

    abs_cont2 = [abs(x) for x in cont2]
    max_val = max(abs_cont2)
    cont2 = [x / max_val for x in cont2]

    X = np.c_[cont1, cont2]

    ica = FastICA(n_components=2, whiten='unit-variance', random_state=42)
    S = ica.fit_transform(X)

    s1 = S[:, 0]
    s2 = S[:, 1]

    abs_s1 = [abs(x) for x in s1]
    max_value_s1 = max(abs_s1)
    s1 = [x / max_value_s1 for x in s1]

    abs_s2 = [abs(x) for x in s2]
    max_value_s2 = max(abs_s2)
    s2 = [x / max_value_s2 for x in s2]

    s1 = np.array(s1)
    s2 = np.array(s2)

    s1_out = (s1 * 32767).astype(np.int16)
    s2_out = (s2 * 32767).astype(np.int16)

    wav.write('separated_1.wav', fs1, s1_out)
    wav.write('separated_2.wav', fs1, s2_out)

    print("Done!")

if __name__ == "__main__":
    seperate_voice()
