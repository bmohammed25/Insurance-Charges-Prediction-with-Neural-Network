📌 Project Overview
This project builds and systematically improves a feedforward neural network to predict medical insurance charges from patient demographic and clinical data.
Starting from a full-batch baseline (R² = 0.23), the project progresses through eight improvement experiments, combining the best-performing configuration from each into a final optimized model (R² = 0.86).
Dataset: Medical Cost Personal Dataset — mirichoi0218 (Kaggle)
Task: Regression — predict continuous insurance charges (USD)
Framework: TensorFlow / Keras
Split: 70% Train · 15% Validation · 15% Test

🎯 Key Results
Baseline vs Final Model
ModelVal MAEVal R²Test R²Full-Batch Baseline$7,859.900.2274—Mini-Batch Baseline (bs=32)$2,921.500.8254—Final Optimized Model$2,919.990.82610.8627
Improvement Summary — All 8 Techniques
TechniqueBest ConfigBest Val R²Batch Sizebs = 160.8264Dropoutrate = 0.10.8229Batch NormalizationBN Layer 1 only0.7892 ⚠️ below baselineEarly Stoppingpatience = 200.8261L1 Regularizationλ = 0.100.8251L2 Regularizationλ = 0.010.8290Learning Ratelr = 0.0050.8437 ✅ best single techniqueOptimizerAdam0.8238

🔬 Notebook Structure
The notebook is organized into 8 sections:
SectionDescriptionSection 1Dataset Preparation — load, inspect, define features and targetSection 2Preprocessing — StandardScaler, OneHotEncoder, feature matrix assemblySection 3Baseline — Full-Batch Training (bs = 936, 1000 epochs)Section 4Baseline — Mini-Batch Training (bs = 32, 500 epochs)Section 5Improvement Experiments (8 techniques, 3 configs each)Section 6Final Model — best hyperparameters combinedSection 7Full Comparison Table + Bar ChartsSection 8Understanding — written explanations for every techniqueConclusionKey findings and final model summary

🗃️ Dataset
Source: Kaggle — Medical Cost Personal Dataset
PropertyDetailsRows1,338 patientsFeatures6 input features + 1 targetTargetcharges — individual medical insurance cost (USD, continuous)Missing valuesNone
Features:
FeatureTypeDescriptionageNumericalAge of the primary beneficiarybmiNumericalBody mass indexchildrenNumericalNumber of dependents coveredsexCategoricalmale / femalesmokerCategoricalyes / noregionCategoricalnortheast / northwest / southeast / southwestchargesTargetIndividual medical costs billed by insurer (USD)
Preprocessing:

Numerical features (age, bmi, children) → StandardScaler (fit on train only)
Categorical features (sex, smoker, region) → OneHotEncoder → 8 binary columns
Final input dimension: 11 features per sample


🧠 Model Architecture
Baseline Architecture (shared across all experiments)
Input(11)
  → Dense(64, ReLU)
  → Dense(32, ReLU)
  → Dense(1, Linear)

Optimizer: Adam (lr = 0.001)
Loss: Mean Squared Error (MSE)
Metrics: MAE, R²
Total params: 2,881

Final Optimized Architecture
Input(11)
  → Dense(64, ReLU) [L2 λ=0.01]
  → Dropout(0.1)
  → Dense(32, ReLU) [L2 λ=0.01]
  → Dropout(0.1)
  → Dense(1, Linear)
HyperparameterValueSourceBatch size16Section 5.1 — best Val R²Learning rate0.005Section 5.7 — best Val R² across all experimentsOptimizerAdamSection 5.8 — best among Adam / RMSprop / SGDL2 regularizationλ = 0.01Section 5.6 — best regularizer testedEarly stoppingpatience = 20Section 5.4 — best Val R²Dropoutrate = 0.1Section 5.2 — lightest effective regularization

📊 Experiment Details
5.1 Batch Size (bs = 16, 32, 64)
Smaller batches produce noisier but more frequent gradient updates, leading to better generalization. bs=16 yields the highest Val R² (0.8264) at the cost of longer training time.
5.2 Dropout (rate = 0.1, 0.2, 0.3)
Rate=0.1 performs best — light regularization prevents co-adaptation without suppressing learning capacity. Higher rates hurt performance on this small dataset.
5.3 Batch Normalization (BN Layer 1, BN Both, BN + Dropout)
All BN configurations fall below the mini-batch baseline. Normalizing already-standardized tabular features removes useful signal. BN was not selected for the final model.
5.4 Early Stopping (patience = 5, 10, 20)
Longer patience consistently yields better results. patience=20 stops at epoch 522, achieving Val R² = 0.8261 while preventing wasted compute.
5.5 L1 Regularization (λ = 0.01, 0.05, 0.1)
Differences between λ values are minimal (< 0.002 R²). L1 encourages sparsity but was outperformed by L2 on this dataset. Not selected for the final model.
5.6 L2 Regularization (λ = 0.001, 0.005, 0.01)
λ=0.01 achieves the best Val R² (0.8290). L2's smooth weight shrinkage is well-suited for regression with correlated features.
5.7 Learning Rate (lr = 0.001, 0.005, 0.01)
lr=0.005 achieves Val R² = 0.8437 — the single highest score across all 8 experiment categories. This is the most impactful single hyperparameter in the study.
5.8 Optimizer (Adam, RMSprop, SGD)
SGD diverges completely (NaN loss) without adaptive scaling. Adam marginally outperforms RMSprop and is selected for the final model.

🔑 Key Findings

Batch size matters more than batch normalization — switching from full-batch (bs=936) to mini-batch (bs=32) raised R² from 0.23 to 0.83, a far larger gain than any individual technique
Learning rate is the single most impactful hyperparameter — lr=0.005 achieved Val R² = 0.8437, the highest score across all 8 experiment categories
Batch normalization hurts on clean tabular data — all BN configurations fell below the mini-batch baseline, confirming BN is most beneficial in deep networks, not shallow regression models on pre-standardized features
SGD diverges without adaptive scaling — NaN loss confirms SGD cannot converge on this task without careful tuning; Adam is the robust default choice
Combining best hyperparameters generalizes well — Final model Test R² (0.8627) exceeds Validation R² (0.8261), confirming no overfitting
Early stopping saves compute — final model stopped at epoch 130 (vs. running all 1,000), training in just 24.93 seconds

📈 Visualizations Included
CategoryVisualizationsBaselinesFull-batch vs mini-batch convergence curves, comparison tableEach ExperimentOverlaid train/validation convergence curves for all 3 configsFinal ModelConvergence curve overlaid with mini-batch baselineSummaryFull comparison table (all configs), 3 bar charts (Val MAE, Val MSE, Val R²)

🙏 Acknowledgements

Dataset: mirichoi0218 — Medical Cost Personal Dataset (Kaggle)
