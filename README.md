## 📌 Project Overview

This project builds and systematically improves a feedforward neural network to predict medical insurance charges from patient demographic and clinical data.

Starting from a full-batch baseline (R² = 0.23), the project progresses through eight improvement experiments, combining the best-performing configuration from each into a final optimized model (R² = 0.86).

**Dataset:** [Medical Cost Personal Dataset — mirichoi0218 (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Task:** Regression — predict continuous insurance charges (USD)  
**Framework:** TensorFlow / Keras  
**Split:** 70% Train · 15% Validation · 15% Test

---

## 🎯 Key Results

### Baseline vs Final Model

| Model | Val MAE | Val R² | Test R² |
|---|---|---|---|
| Full-Batch Baseline | $7,859.90 | 0.2274 | — |
| Mini-Batch Baseline (bs=32) | $2,921.50 | 0.8254 | — |
| **Final Optimized Model** | **$2,919.99** | **0.8261** | **0.8627** |

### Improvement Summary — All 8 Techniques

| Technique | Best Config | Best Val R² |
|---|---|---|
| Batch Size | bs = 16 | 0.8264 |
| Dropout | rate = 0.1 | 0.8229 |
| Batch Normalization | BN Layer 1 only | 0.7892 ⚠️ below baseline |
| Early Stopping | patience = 20 | 0.8261 |
| L1 Regularization | λ = 0.10 | 0.8251 |
| L2 Regularization | λ = 0.01 | 0.8290 |
| Learning Rate | lr = 0.005 | **0.8437** ✅ best single technique |
| Optimizer | Adam | 0.8238 |

---
## 🔬 Notebook Structure

The notebook is organized into 8 sections:

| Section | Description |
|---|---|
| Section 1 | Dataset Preparation — load, inspect, define features and target |
| Section 2 | Preprocessing — StandardScaler, OneHotEncoder, feature matrix assembly |
| Section 3 | Baseline — Full-Batch Training (bs = 936, 1000 epochs) |
| Section 4 | Baseline — Mini-Batch Training (bs = 32, 500 epochs) |
| Section 5 | Improvement Experiments (8 techniques, 3 configs each) |
| Section 6 | Final Model — best hyperparameters combined |
| Section 7 | Full Comparison Table + Bar Charts |
| Section 8 | Understanding — written explanations for every technique |
| Conclusion | Key findings and final model summary |

---

## 🗃️ Dataset

**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

| Property | Details |
|---|---|
| Rows | 1,338 patients |
| Features | 6 input features + 1 target |
| Target | `charges` — individual medical insurance cost (USD, continuous) |
| Missing values | None |

**Features:**

| Feature | Type | Description |
|---|---|---|
| age | Numerical | Age of the primary beneficiary |
| bmi | Numerical | Body mass index |
| children | Numerical | Number of dependents covered |
| sex | Categorical | male / female |
| smoker | Categorical | yes / no |
| region | Categorical | northeast / northwest / southeast / southwest |
| **charges** | **Target** | **Individual medical costs billed by insurer (USD)** |

**Preprocessing:**
- Numerical features (`age`, `bmi`, `children`) → StandardScaler (fit on train only)
- Categorical features (`sex`, `smoker`, `region`) → OneHotEncoder → 8 binary columns
- Final input dimension: **11 features** per sample

---

## 🧠 Model Architecture

### Baseline Architecture (shared across all experiments)

```
Input(11)
  → Dense(64, ReLU)
  → Dense(32, ReLU)
  → Dense(1, Linear)
```

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** MAE, R²
- **Total params:** 2,881

### Final Optimized Architecture

```
Input(11)
  → Dense(64, ReLU) [L2 λ=0.01]
  → Dropout(0.1)
  → Dense(32, ReLU) [L2 λ=0.01]
  → Dropout(0.1)
  → Dense(1, Linear)
```

| Hyperparameter | Value | Source |
|---|---|---|
| Batch size | 16 | Section 5.1 — best Val R² |
| Learning rate | 0.005 | Section 5.7 — best Val R² across all experiments |
| Optimizer | Adam | Section 5.8 — best among Adam / RMSprop / SGD |
| L2 regularization | λ = 0.01 | Section 5.6 — best regularizer tested |
| Early stopping | patience = 20 | Section 5.4 — best Val R² |
| Dropout | rate = 0.1 | Section 5.2 — lightest effective regularization |

---

## 📊 Experiment Details

### 5.1 Batch Size (bs = 16, 32, 64)
Smaller batches produce noisier but more frequent gradient updates, leading to better generalization. bs=16 yields the highest Val R² (0.8264) at the cost of longer training time.

### 5.2 Dropout (rate = 0.1, 0.2, 0.3)
Rate=0.1 performs best — light regularization prevents co-adaptation without suppressing learning capacity. Higher rates hurt performance on this small dataset.

### 5.3 Batch Normalization (BN Layer 1, BN Both, BN + Dropout)
All BN configurations fall **below** the mini-batch baseline. Normalizing already-standardized tabular features removes useful signal. BN was not selected for the final model.

### 5.4 Early Stopping (patience = 5, 10, 20)
Longer patience consistently yields better results. patience=20 stops at epoch 522, achieving Val R² = 0.8261 while preventing wasted compute.

### 5.5 L1 Regularization (λ = 0.01, 0.05, 0.1)
Differences between λ values are minimal (< 0.002 R²). L1 encourages sparsity but was outperformed by L2 on this dataset. Not selected for the final model.

### 5.6 L2 Regularization (λ = 0.001, 0.005, 0.01)
λ=0.01 achieves the best Val R² (0.8290). L2's smooth weight shrinkage is well-suited for regression with correlated features.

### 5.7 Learning Rate (lr = 0.001, 0.005, 0.01)
lr=0.005 achieves **Val R² = 0.8437** — the single highest score across all 8 experiment categories. This is the most impactful single hyperparameter in the study.

### 5.8 Optimizer (Adam, RMSprop, SGD)
SGD diverges completely (NaN loss) without adaptive scaling. Adam marginally outperforms RMSprop and is selected for the final model.

---

## 🔑 Key Findings

- **Batch size matters more than batch normalization** — switching from full-batch (bs=936) to mini-batch (bs=32) raised R² from 0.23 to 0.83, a far larger gain than any individual technique
- **Learning rate is the single most impactful hyperparameter** — lr=0.005 achieved Val R² = 0.8437, the highest score across all 8 experiment categories
- **Batch normalization hurts on clean tabular data** — all BN configurations fell below the mini-batch baseline, confirming BN is most beneficial in deep networks, not shallow regression models on pre-standardized features
- **SGD diverges without adaptive scaling** — NaN loss confirms SGD cannot converge on this task without careful tuning; Adam is the robust default choice
- **Combining best hyperparameters generalizes well** — Final model Test R² (0.8627) exceeds Validation R² (0.8261), confirming no overfitting
- **Early stopping saves compute** — final model stopped at epoch 130 (vs. running all 1,000), training in just 24.93 seconds

---

## 📈 Visualizations Included

| Category | Visualizations |
|---|---|
| Baselines | Full-batch vs mini-batch convergence curves, comparison table |
| Each Experiment | Overlaid train/validation convergence curves for all 3 configs |
| Final Model | Convergence curve overlaid with mini-batch baseline |
| Summary | Full comparison table (all configs), 3 bar charts (Val MAE, Val MSE, Val R²) |

---


## 🙏 Acknowledgements

- Dataset: [mirichoi0218 — Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
