# MAGIC Gamma Telescope Particle Classification

An end-to-end machine learning project classifying simulated atmospheric Cherenkov gamma rays versus hadronic background showers using the **MAGIC Gamma Telescope Dataset**. The project covers data preprocessing, feature normalization, class balancing using random oversampling, and supervised classification using **Gaussian Naive Bayes** and **k-Nearest Neighbors (k-NN)**.

---

## 📌 Project Overview
- **Dataset:** MAGIC Gamma Telescope Data (`magic04.data`)[cite: 2]
- **Target:** Binary classification of simulated Cherenkov shower images:
  - `g` (Gamma rays) $\rightarrow$ Assigned to `1` (Signal)[cite: 2]
  - `h` (Hadronic background) $\rightarrow$ Assigned to `0` (Noise)[cite: 2]
- **Total Records:** 19,020 instances across 10 continuous numeric features and 1 target label[cite: 2].
- **Core Focus:** Handling class imbalance via oversampling and evaluating probabilistic versus distance-based classifiers[cite: 2].

---

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.x[cite: 2]
- **Data Manipulation:** `pandas`, `numpy`[cite: 2]
- **Visualization:** `matplotlib`[cite: 2]
- **Preprocessing & Balancing:** `scikit-learn` (`StandardScaler`), `imbalanced-learn` (`RandomOverSampler`)[cite: 2]
- **Modeling & Evaluation:** `sklearn.naive_bayes.GaussianNB`, `sklearn.neighbors.KNeighborsClassifier`, `sklearn.metrics.classification_report`[cite: 2]

---

## ⚙️ Machine Learning Pipeline

1. **Data Ingestion & Schema Assignment:**
   - Load raw `.data` format and map the 11 feature names (`fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`, `class`)[cite: 2].
2. **Target Encoding:**
   - Convert categorical classes (`g`/`h`) into integer labels (`1`/`0`)[cite: 2].
3. **Stratified Split Strategy:**
   - Shuffled split partitioned into:
     - **Train:** 60% (~11,412 samples)[cite: 2]
     - **Validation:** 20% (~3,804 samples)[cite: 2]
     - **Test:** 20% (~3,804 samples)[cite: 2]
4. **Feature Scaling & Resampling:**
   - Standardized features with `StandardScaler` to ensure zero mean and unit variance[cite: 2].
   - Applied `RandomOverSampler` on the training partition to balance class representation (7,398 samples per class)[cite: 2].
5. **Model Training & Inference:**
   - **Gaussian Naive Bayes (`GaussianNB`)**: Fast probabilistic baseline applying Bayes' theorem with assumed Gaussian feature distributions[cite: 2].
   - **k-Nearest Neighbors (`KNeighborsClassifier`)**: Non-parametric baseline using $k=3$ with Minkowski distance metrics[cite: 2].

---

## 📊 Dataset Features

| Feature | Type | Description |
| :--- | :--- | :--- |
| `fLength` | Continuous | Major axis of ellipse [mm][cite: 2] |
| `fWidth` | Continuous | Minor axis of ellipse [mm][cite: 2] |
| `fSize` | Continuous | 10-log of sum of content of all photomultipliers [photons][cite: 2] |
| `fConc` | Continuous | Ratio of sum of two highest pixels over total sum[cite: 2] |
| `fConc1` | Continuous | Ratio of highest pixel over total sum[cite: 2] |
| `fAsym` | Continuous | Distance of highest pixel to center, projected onto major axis[cite: 2] |
| `fM3Long` | Continuous | 3rd root of 3rd moment along major axis[cite: 2] |
| `fM3Trans` | Continuous | 3rd root of 3rd moment along minor axis[cite: 2] |
| `fAlpha` | Continuous | Angle of major axis with vector to camera center [deg][cite: 2] |
| `fDist` | Continuous | Distance from camera center to shower center[cite: 2] |
| `class` | Binary | `1` for Gamma event (`g`), `0` for Hadron event (`h`)[cite: 2] |

---

## 📈 Model Performance & Evaluation

### Gaussian Naive Bayes Performance (Validation Split)[cite: 2]
```text
              precision    recall  f1-score   support

         0.0       0.35      0.22      0.27      1333
         1.0       0.65      0.79      0.71      2471

    accuracy                           0.59      3804
   macro avg       0.50      0.50      0.49      3804
weighted avg       0.55      0.59      0.56      3804
