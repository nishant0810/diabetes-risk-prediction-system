# Diabetes Risk Prediction System

A Machine Learning project that predicts whether a person is likely to have diabetes using medical diagnostic data. This project uses data preprocessing, feature scaling, model training, evaluation, and visualization techniques to compare the performance of multiple classification algorithms.

---

## Features

* Data preprocessing and cleaning
* Handling missing/invalid values
* Feature scaling using `StandardScaler`
* Training multiple ML models:

  * Logistic Regression
  * Random Forest Classifier
* Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix
* Feature importance visualization

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Dataset

The project uses the `diabetes.csv` dataset based on the PIMA Indians Diabetes Database.

### Features in Dataset

| Feature                  | Description                       |
| ------------------------ | --------------------------------- |
| Pregnancies              | Number of pregnancies             |
| Glucose                  | Plasma glucose concentration      |
| BloodPressure            | Diastolic blood pressure          |
| SkinThickness            | Triceps skin fold thickness       |
| Insulin                  | 2-Hour serum insulin              |
| BMI                      | Body mass index                   |
| DiabetesPedigreeFunction | Diabetes pedigree function        |
| Age                      | Age of patient                    |
| Outcome                  | Diabetes result (0 = No, 1 = Yes) |

---

## Project Workflow

### 1. Import Libraries

The required libraries for data handling, preprocessing, visualization, and machine learning are imported.

### 2. Load Dataset

```python
df = pd.read_csv("diabetes.csv")
```

### 3. Data Cleaning

Some medical features contain `0` values which are biologically invalid. These values are replaced with `NaN` and filled using median values.

```python
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[cols] = df[cols].replace(0, np.nan)

df.fillna(df.median(), inplace=True)
```

### 4. Feature Selection

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```

### 5. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 6. Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 7. Model Training

#### Logistic Regression

```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

#### Random Forest Classifier

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

---

## Model Evaluation

The models are evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* Confusion Matrix

```python
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)
```

---

## Visualizations

### Confusion Matrix

Heatmaps are generated using Seaborn to visualize model predictions.

### Feature Importance

Random Forest feature importance helps identify which medical factors contribute most to diabetes prediction.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/diabetes-risk-prediction-system.git
```

### 2. Navigate to Project Folder

```bash
cd diabetes-risk-prediction-system
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 4. Run the Script

```bash
python main.py
```

---

## Expected Output

* Model accuracy metrics
* Precision and recall values
* Confusion matrix plots
* Feature importance graph

---

## Future Improvements

* Add XGBoost and SVM models
* Hyperparameter tuning
* Deploy using Flask or Streamlit
* Add real-time prediction UI
* Improve dataset balancing

---

## Learning Outcomes

This project demonstrates:

* Data preprocessing techniques
* Machine Learning classification workflow
* Model comparison and evaluation
* Data visualization
* Feature importance analysis

---

## License

This project is open-source and available under the MIT License.
