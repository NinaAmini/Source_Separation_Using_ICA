# Audio Source Separation via Independent Component Analysis (ICA)

This repository contains a Python-based implementation of **Blind Source Separation (BSS)**. Using the **FastICA** algorithm, the project demonstrates how to decouple mixed audio signals (the "Cocktail Party Problem") into their original, independent constituent sources.


## 🛠 Project Overview

In this implementation, two microphones (sensors) record a linear combination of two independent sound sources. The goal is to recover the source signals $S$ from the observed mixtures $X$ without prior knowledge of the mixing matrix $A$.

### Key Technical Features:
* **Signal Preprocessing:** Automated stereo-to-mono downmixing via channel averaging.
* **Temporal Alignment:** Dynamic truncation to ensure input vector compatibility ($L_1 = L_2$).
* **Normalization Pipeline:** Dual-stage peak normalization to prevent digital clipping.
* **Statistical De-mixing:** Leverages `scikit-learn`'s FastICA with unit-variance whitening.
* **Encoding:** High-fidelity conversion from `float32` to `int16` PCM for WAV output.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. You can install the necessary signal processing and machine learning libraries using:

```bash
pip install numpy scipy scikit-learn matplotlib

