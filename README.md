
<h1 align="center">🌊 Probabilistic Forecasting of Sea Level Anomaly &amp;<br/>Significant Wave Height over the Indian Ocean</h1>

<p align="center">
  <strong>9 architectures · 2 loss functions · 5 ocean locations · 2 variables · 900+ training runs</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&amp;logoColor=white" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&amp;logoColor=white" alt="TensorFlow"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Keras-Custom_Losses-D00000?logo=keras&amp;logoColor=white" alt="Keras"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Published-success" alt="Status"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Training_Runs-900+-blueviolet" alt="Runs"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License"/></a>
</p>

<p align="center">
  <a href="#-key-results">Results</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-models">Models</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📌 TL;DR

> We trained **9 deep learning architectures** under **2 probabilistic loss functions** to predict calibrated 95% prediction intervals for **Sea Level Anomaly (SLA)** and **Significant Wave Height (SWH)** at 5 Indian Ocean locations. Over **900+ training runs** (5 seeds each) reveal that:
>
> - **GRU-Quantile** is the globally optimal SLA model (mean WIS = 0.01278, PICP ≥ 95% at 4/5 locations)
> - **SimpleRNN-Quantile** is the only model passing 95% at ALL 5 SLA locations
> - **MDN-GRU-K3** dominates SWH forecasting (best at 3/5 locations)
> - **Transformer &amp; Mamba catastrophically fail on SLA** (PICP as low as 39.1%, CWC = 2.67×10⁹) but **recover on SWH** (95.5%)
>
> This **architecture-variable interaction** — the central finding — is explained by three mechanisms: signal amplitude, forcing regularity, and parameter-to-data ratio.

---

## 🎯 Research Objectives

1. **Benchmark** 9 architectures under identical preprocessing, splitting, and evaluation conditions for probabilistic interval estimation
2. **Discover** why certain architectures fail on SLA but recover on SWH (the architecture-variable interaction)
3. **Identify** Pareto-optimal models that are not dominated in both coverage (PICP) and sharpness (WIS)
4. **Quantify** the real-world decision-theoretic value of model quality (₹45 crore/year difference for port operations)

---

## 🏆 Key Results

### SLA — Global Model Ranking

| Rank | Model | Mean PICP (%) | Mean WIS | Mean MPIW (m) | Pass 5/5? |
|:----:|-------|:-------------:|:--------:|:-------------:|:---------:|
| 🥇 | **GRU-Quantile** | 96.7 | **0.01374** | **0.01142** | 4/5 |
| 🥈 | SimpleRNN-Quantile | 97.0 | 0.01378 | 0.01284 | **5/5** ✅ |
| 🥉 | GRU-Tube | 97.4 | 0.01714 | 0.01342 | 5/5 ✅ |
| 4 | SimpleRNN-Tube | 98.5 | 0.01759 | 0.01398 | 5/5 ✅ |
| 5 | DeepAR | 95.5 | 0.01984 | 0.01760 | 4/5 |
| 6 | MDN-GRU-K3 | 99.3 | 0.02506 | 0.02471 | 5/5 ✅ |
| ⚠️ | Transformer | 65.1–76.4 | 0.33–0.41 | — | 0/5 ❌ |
| ⚠️ | Mamba | 73.7–91.0 | 0.20–0.26 | — | 0–1/5 ❌ |

### The Central Discovery

```
Same Transformer architecture:
  SLA (Southern IO)  → PICP = 46.6%  | CWC = 2.67 × 10⁹  ❌ Catastrophic failure
  SWH (Andaman Sea)  → PICP = 95.5%  | CWC = 0.899        ✅ Near-perfect

Three mechanisms:
  1. SWH signal is 20× stronger than SLA (metres vs centimetres)
  2. SWH has a dominant annual monsoon cycle; SLA has competing multi-scale dynamics
  3. 90K parameters / 846 training sequences = severe overfitting for subtle SLA signal
```

---

## 🏗️ Architecture

### System Pipeline

```
CMEMS Satellite Data (2021–2023, 1096 daily obs)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  RobustScaler (median + IQR)
│   NaN interpolation │  80/20 temporal split (877/219 days)
│   SEQ_LEN = 30      │  Sliding window → 846 training sequences
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              9 Model Architectures              │
│                                                 │
│  Recurrent:  GRU │ LSTM │ SimpleRNN             │
│  Attention:  Transformer (4-head, 90K params)   │
│  SSM:        Mamba (selective, 90K params)       │
│  Generative: DeepAR (Gaussian likelihood)       │
│  Mixture:    MDN-GRU-K3 (3-component GMM)       │
│  Hybrid:     LSTM-KDE (KDE + conformal)         │
│  Classical:  ARIMA(2,1,1) (bootstrap CI)        │
│                                                 │
│  × 2 Loss Functions: Quantile │ Tube            │
│  × 5 Random Seeds: {42, 7, 13, 99, 2025}       │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│    Dense(32, ReLU)  │  Shared backbone
│    Dense(2, linear) │  Output = [lower_bound, upper_bound]
│    Dropout(0.2)     │  L2 = 1e-4
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│           Evaluation (4 Metrics)                │
│                                                 │
│  PICP  ─  Coverage probability (target ≥ 95%)   │
│  MPIW  ─  Interval width (lower = sharper)      │
│  WIS   ─  Winkler Score (primary ranking)       │
│  CWC   ─  Exp. penalty for under-coverage       │
└─────────────────────────────────────────────────┘
```

---

## 🧮 Loss Functions

### Quantile (Pinball) Loss — Winner on WIS

```
ρ_q(e) = max(q·e, (q-1)·e)

q=0.10 → under-prediction penalised 9× more → learns 10th percentile
q=0.90 → over-prediction penalised 9× more → learns 90th percentile
```

### Tube Loss (Prof. Anand, [arXiv:2412.06853](https://arxiv.org/abs/2412.06853))

```
L = Σ coverage_penalty(y, μ₁, μ₂) + δ · |μ₂ - μ₁|
δ = 0.01 (SLA) / 0.03 (SWH)
```

---

## 📊 Evaluation Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **PICP** | % of test points inside the predicted interval | ≥ 95% |
| **MPIW** | Average interval width (metres) | Lower = better |
| **WIS** | Joint coverage + sharpness score | Lower = better (primary) |
| **CWC** | MPIW with exponential under-coverage penalty | Lower = better |

---

## 📂 Dataset

| Variable | Source | Grid | Period |
|----------|--------|------|--------|
| **SLA** | CMEMS `SEALEVEL_GLO_PHY_L4_MY_008_047` | 0.125° | 2021–2023 |
| **SWH** | CMEMS Wave model (`VHM0`) | 0.2° | 2021–2023 |

**5 locations:** Arabian Sea · Bay of Bengal · Andaman Sea · Lakshadweep · Southern IO

---

## 🚀 Quick Start

```bash
git clone https://github.com/Bhavya-2805/Ocean-Probabilistic-Forecasting.git
cd Ocean-Probabilistic-Forecasting
pip install -r requirements.txt

# Train GRU-Quantile on Arabian Sea SLA
python scripts/run_full_benchmark.py --model gru --loss quantile --location Arabian_Sea

# Generate all 10 comparative analysis figures
python comparative_analysis.py
```

---

## 🧠 Models — Parameter Counts &amp; Training Times

| Architecture | Params | Ratio | Time | SLA Rank |
|-------------|:------:|:-----:|:----:|:--------:|
| SimpleRNN | ~18K | 21 | 6:11 | #2 |
| **GRU** | **~33K** | **39** | **9:43** | **#1** |
| LSTM | ~44K | 52 | 8:53 | #6 |
| DeepAR | ~33K | 39 | 4:01 | #5 |
| MDN-GRU | ~35K | 41 | 3:46 | #7 |
| Transformer | ~90K | 107 | — | #13–14 |
| Mamba | ~90K | 107 | — | #11–12 |

> Times on Apple M5 (CPU/MPS). Ratio = params / training sequences.

---

## 💡 Key Findings

1. **Architecture-Variable Interaction (Novel):** Same Transformer = 46.6% on SLA → 95.5% on SWH
2. **Parameter Efficiency > Capacity:** GRU (33K) beats Transformer (90K) on small ocean data
3. **Quantile Loss Wins on WIS:** 15–30% narrower intervals than Tube Loss
4. **SWH Calibration is More Uniform:** 61% pairs ≥95% PICP (vs 51% SLA)
5. **Real-World Impact:** ₹45 crore/year advantage for GRU-Q over ARIMA in port operations

---

## 📖 References

1. Anand et al. (2024). "Tube Loss." [arXiv:2412.06853](https://arxiv.org/abs/2412.06853)
2. Hochreiter &amp; Schmidhuber (1997). "LSTM." *Neural Computation* 9(8)
3. Cho et al. (2014). "GRU." EMNLP
4. Vaswani et al. (2017). "Attention Is All You Need." NeurIPS
5. Gu &amp; Dao (2023). "Mamba." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
6. Zeng et al. (2023). "Are Transformers Effective for Time Series?" AAAI
7. Winkler (1972). "Interval Estimation." *JASA* 67(337)

---

## 📝 Citation

```bibtex
@misc{
  title   = {Probabilistic Forecasting of SLA and SWH over the Indian Ocean},
  author  = {Tirth Gandhi, Bhavya Boda},
  year    = {2026},
  school  = {Dhirubhai Ambani University},
  note    = {B.Tech Mini Project, PC406. Faculty: Prof. Pritam Anand}
}
```

---

## 👥 Authors

| | Name | ID | Role |
|---|------|----|----|
| 🧑‍💻 | **Tirth Gandhi** | 202301016 | B.Tech ICT, DA-IICT |
| 🧑‍💻 | **Bhavya Boda** | 202301037 | B.Tech ICT, DA-IICT |
| 👨‍🏫 | **Prof. Pritam Anand** | Faculty Mentor | Creator of Tube Loss |

---

<p align="center">
  <strong>⭐ Star this repo if you find it useful!</strong>
</p>
