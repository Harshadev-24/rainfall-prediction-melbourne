
# 🌧️ Rain Prediction Model for Melbourne Area

A machine learning classifier that predicts daily rainfall in the Melbourne metropolitan area using historical Australian weather data. Built as the capstone project in Machine Learning with Python for IBM AI Engineering Professional Certificate.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

## 🎯 Project Overview

This project builds a binary classification model to predict whether it will rain today based on historical weather observations from three locations in the Melbourne area: Melbourne, Melbourne Airport, and Watsonia. The model uses 22 meteorological features to make predictions with high accuracy.

**Key Highlights:**
- Predicts rainfall with ~85% accuracy
- Implements multiple ML algorithms (Random Forest, Logistic Regression)
- Uses scikit-learn pipelines for reproducible workflows
- Handles imbalanced data with stratified sampling
- Optimized with GridSearchCV hyperparameter tuning

## 📊 Dataset

**Source:** [Australian Government Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) via [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

**Time Period:** 2008-2017 (10 years)

**Locations:** Melbourne, Melbourne Airport, Watsonia (15-18 km radius)

**Final Dataset Size:** 7,557 observations after preprocessing

### Features (22 total)

**Numerical Features (16):**
- Temperature: MinTemp, MaxTemp, Temp9am, Temp3pm
- Atmospheric: Pressure9am, Pressure3pm, Humidity9am, Humidity3pm
- Wind: WindSpeed9am, WindSpeed3pm, WindGustSpeed
- Weather: Rainfall, Evaporation, Sunshine, Cloud9am, Cloud3pm

**Categorical Features (6):**
- Location: Melbourne, Melbourne Airport, Watsonia
- Wind Direction: WindGustDir, WindDir9am, WindDir3pm
- Season: Spring, Summer, Autumn, Winter (engineered feature)
- RainYesterday: Yes/No

**Target Variable:** RainToday (Yes/No)

## ✨ Features

### Data Preprocessing
- ✅ Removed rows with missing values (retained 56k→7.5k complete records)
- ✅ Feature engineering: extracted Season from Date
- ✅ Addressed data leakage by reframing target variable
- ✅ Localized analysis to Melbourne metropolitan area

### Model Pipeline
- ✅ **ColumnTransformer** for separate numerical/categorical preprocessing
- ✅ **StandardScaler** for numerical feature normalization
- ✅ **OneHotEncoder** for categorical variable encoding
- ✅ **StratifiedKFold** cross-validation (5 folds)
- ✅ **GridSearchCV** for hyperparameter optimization

### Machine Learning Models
1. **Random Forest Classifier**
   - Best params: n_estimators=100, max_depth=None, min_samples_split=2

2. **Logistic Regression**
   - Regularization: C=0.1, penalty='l2', solver='lbfgs'

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rainfall-prediction-melbourne.git
cd rainfall-prediction-melbourne
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### requirements.txt
```
numpy>=1.23
pandas>=1.2
matplotlib>=3.4
scikit-learn>=1.2
seaborn>=0.13
jupyter>=1.0
```

## 💻 Usage

### Running the Notebook

1. **Launch Jupyter Notebook**
```bash
jupyter notebook FinalProject_AUSWeather.ipynb
```

2. **Execute cells sequentially** or run all cells:
   - Cell > Run All

### Using the Trained Model

```python
from sklearn.pipeline import Pipeline
import pandas as pd

# Load the notebook and extract fitted model
# Or train a new model using the pipeline

# Make predictions
sample_data = {
    'Location': 'Melbourne',
    'MinTemp': 12.5,
    'MaxTemp': 20.3,
    # ... other features
}

prediction = gridsearch.predict(pd.DataFrame([sample_data]))
print(f"Rain today: {prediction[0]}")
```

## 📈 Model Performance

### Random Forest (Best Model)

| Metric | No Rain | Rain | Weighted Avg |
|--------|---------|------|--------------|
| **Precision** | 0.87 | 0.75 | 0.84 |
| **Recall** | 0.93 | 0.59 | 0.85 |
| **F1-Score** | 0.90 | 0.66 | 0.84 |

**Overall Accuracy:** 85%

**Confusion Matrix:**
- True Negatives: 1071
- False Positives: 82
- False Negatives: 147
- True Positives: 212

### Logistic Regression

| Metric | Value |
|--------|-------|
| **Accuracy** | 82% |
| **Precision (Rain)** | 0.68 |
| **Recall (Rain)** | 0.54 |

### Key Insights
- **Class Imbalance:** 76% no-rain days vs 24% rain days
- **Model Strength:** Excellent at predicting non-rain days (93% recall)
- **Challenge:** Lower recall for rain prediction (59%) due to class imbalance
- **Feature Importance:** Humidity3pm, Pressure3pm, and Cloud3pm are top predictors

## 📁 Project Structure

```
rainfall-prediction-melbourne/
│
├── FinalProject_AUSWeather.ipynb   # Main Jupyter notebook
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
│
├── data/                            # Dataset (optional - if small)
│   └── weatherAUS.csv
│
└── images/                          # Visualizations
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── class_distribution.png

```

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **ML Framework** | scikit-learn 1.2+ |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

## 🎓 Key Learnings

1. **Data Leakage Awareness:** Reframed the prediction target to avoid using same-day features
2. **Pipeline Architecture:** Built modular, reproducible ML workflows
3. **Imbalanced Data Handling:** Used stratified sampling and evaluated beyond accuracy
4. **Feature Engineering:** Created seasonal features from timestamps
5. **Hyperparameter Tuning:** Optimized models with GridSearchCV
6. **Model Interpretation:** Analyzed feature importances and confusion matrices

## 🔮 Future Improvements

- [ ] Implement SMOTE or class weighting for better recall on rain days
- [ ] Add more recent data (2018-2025) for improved predictions
- [ ] Build an ensemble model combining Random Forest + XGBoost
- [ ] Create an interactive web app using Streamlit
- [ ] Expand to predict rainfall amount (regression problem)
- [ ] Include additional weather stations across Australia
- [ ] Deploy model as REST API using Flask/FastAPI

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/Harshadev-24)
- LinkedIn: [Your Profile](http://www.linkedin.com/in/harsha-vardhan-847296257)
- Email: harsha2003hvd@gmail.com

## 🙏 Acknowledgments

- **IBM AI Engineering Professional Certificate - Machine Learning with Python** for the project framework
- **Australian Bureau of Meteorology** for the dataset
- **Kaggle** for hosting the data
- **scikit-learn** community for excellent documentation

---

⭐ If you found this project helpful, please consider giving it a star!

**Certificate:** [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer)
