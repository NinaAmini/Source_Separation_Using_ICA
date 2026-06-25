# 🎧 Audio Source Separation using Independent Component Analysis (ICA)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![ML](https://img.shields.io/badge/ML-FastICA-orange)
![Status](https://img.shields.io/badge/status-active-success)

---

## 📌 Overview

This project implements **Blind Source Separation (BSS)** using **Independent Component Analysis (ICA)** to solve the *Cocktail Party Problem* — separating mixed audio signals into original independent sources.

We assume:
\[
X = A S
\]
and estimate \( S \) without knowing \( A \).

---

## ⚙️ Pipeline

```mermaid
graph TD
A[Audio WAVs] --> B[Mono Conversion]
B --> C[Length Alignment]
C --> D[Normalization]
D --> E[FastICA]
E --> F[Signal Reconstruction]
F --> G[WAV Output]
