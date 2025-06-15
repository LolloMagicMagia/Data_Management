# Data Management for Machine Learning – Asteroids Dataset

**Data Architecture Project – University of Milano-Bicocca**  
**Academic Year:** 2023–2024  
**Final Grade:** 27/30

Team Members:
- Mattia Biancini – ID 865966
- Marco Gerardi – ID 869138
- Lorenzo Monti – ID 869960
---

## Overview

This project focuses on **data quality assessment**, **preprocessing**, and **feature impact analysis** in the context of a Machine Learning task based on a dataset of Near-Earth Objects (asteroids), provided by [NASA's NEOWS API](https://api.nasa.gov/).

The primary objectives were:

- Data cleaning and Principal Component Analysis (PCA)
- Training ML models and evaluating performance using F1 Score
- Artificially "corrupting" the dataset to simulate poor data quality
- Comparing clean vs. corrupted datasets to assess the impact of specific features
- Understanding feature importance from the model’s perspective

---

## Project Structure

- **`input/`** – Contains raw and intermediate datasets
- **`models/`** – Stores trained models serialized with `pickle`
- **`notebooks/`** – Jupyter notebooks used for exploration, EDA, and testing
- **`src/`** – Core Python scripts
  - `train.py` – Trains the model selected and saves it
  - `predict.py` – Loads a trained model and makes predictions
  - `model_selection.py` – Identifies the best performing model
  - `tune_model.py` – Hyperparameter optimization pipeline
  - `utils.py` – Helper functions for preprocessing, validation, etc.

---

## Dataset Information

The dataset is related to asteroids and contains orbital and physical parameters.

- **Instances:** 4687
- **Features:** 40
- **Missing values:** None (in original dataset)

🔗 **Original ML Project using this dataset:**  
[Link to ML Project Repository](https://github.com/LolloMagicMagia/ML-Laboratory)

📓 [Source of the dataset](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification)

---

## Feature Examples

- `Neo Reference ID`: Unique asteroid identifier
- `Absolute Magnitude`: Intrinsic brightness
- `Estimated Diameter (km/m)`: Physical size of the object
- `Relative Velocity`: In km/s or km/h
- `Jupiter Tisserand Invariant`: Orbital dynamic parameter
- `Eccentricity`, `Semi-Major Axis`, `Orbital Period`
- `Hazardous`: Boolean classification label

---

## Project Phases

### 1. **Problem Definition & Goals**
- Binary classification: Predict whether an asteroid is hazardous
- Evaluation metric: **F1 Score** (due to class imbalance)

### 2. **Data Collection and Cleaning**
- Loaded data from NEOWS
- Removed irrelevant or redundant features
- Checked for duplicates, consistency, and logical errors

### 3. **Exploratory Data Analysis (EDA)**
- Visualizations: Histograms, boxplots, scatterplots
- Correlation matrix
- Descriptive statistics: mean, median, variance

### 4. **Feature Engineering**
- **Principal Component Analysis (PCA)**: Reduced dimensionality
- Created meaningful derived features
- Standardization and normalization applied where necessary

### 5. **Model Training (Clean Dataset)**
- Trained various models (e.g., Random Forest, SVM, Logistic Regression)
- Performed cross-validation
- Evaluated using F1 Score

### 6. **Data Corruption and Degradation**
Simulated violations of **Data Quality Principles**:

- Injected missing values (100% in one or more features)
- Introduced outliers and inconsistent values based on IQRs
- Deleted features entirely to simulate information loss

### 7. **Model Training (Corrupted Dataset)**
- Retrained the same models using degraded data
- Measured performance drop using F1 Score

### 8. **Comparison and Interpretation**
- Quantitative comparison between clean and corrupted model performances
- Identified most impactful features based on how their corruption degraded results
- Highlighted the **critical role of data quality** in model accuracy and reliability

---

## Key Takeaways

- PCA helped simplify the dataset without significant loss of information
- Certain features were shown to be **highly influential** on the model’s outcome
- Data degradation clearly highlighted the **sensitivity of ML models** to input quality
- F1 Score proved effective in assessing performance with unbalanced classes

---

## Conclusion

This project emphasizes the crucial importance of **data preprocessing and quality** in any machine learning pipeline. Through a controlled experiment involving both clean and manipulated data, we quantified how different types of degradation influence model performance and drew conclusions about which features are most valuable for predicting asteroid hazards.

---

📘 *Full project documentation and code are available in the repository.*
