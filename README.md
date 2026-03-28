# 🏡 House Price Prediction 

## 📽️ Project Overview
This project builds a machine‑learning system that predicts Boston house prices using the classic housing dataset from CMU/Scikit‑learn.
It walks through the full workflow: 
- Loading and reconstructing the dataset,
- Cleaning,
- Exploring correlations,
- Splitting into train/test sets,
- Training regression models (XGBoost & Decision Tree),
- Evaluating performance, and
- Tuning hyperparameters.

The goal is to understand which features influence housing prices and to build accurate, generalizable prediction models for real‑estate decision‑making


### 🧰 1. Setup & Dependencies
#### Install required libraries:
```
pip install scikit-learn xgboost imbalanced-learn pandas numpy matplotlib seaborn
```
Imported libraries include:
- `pandas`,  `numpy`
- `matplotlib`,  `seaborn`
- `sklearn` (train_test_split, metrics, DecisionTreeRegressor, GridSearchCV)
- `xgboost` (XGBClassifier)

### 📥 2. Data Sourcing
The dataset is loaded from the CMU repository:
```
raw_df = pd.read_csv("http://lib.stat.cmu.edu/datasets/boston", ...)
```
Because the Boston dataset is stored in an alternating‑row format, the notebook:
- Reconstructs the feature matrix and target values
- Assigns human‑readable column names
- Combines everything into a clean DataFrame
This ensures the features align correctly with the target (`price`)


### 🧹 3. Data Preprocessing
#### 3.1 Missing Values
Missing values were checked and none required imputation. Rows with missing entries were dropped when necessary.

#### 3.2 Duplicate Values
Duplicates were identified and removed.
```
df_clean = df.drop_duplicates()
```

#### 3.3 Correlation Analysis
A heatmap was generated to visualize relationships between features and price.
This helps identify strong predictors such as:
- RM (average rooms per dwelling)
- LSTAT (% lower‑status population)
- PTRATIO (pupil‑teacher ratio)


### 🔀 4. Train–Test Split
The dataset was split into training and testing sets:
```
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
```
A held‑out test set ensures unbiased evaluation and reproducibility.


### 🤖 5. Model Training
Two regression models were trained:

#### 5.1 XGBoost Regressor
A powerful gradient‑boosted tree model capable of capturing complex, non‑linear patterns.
```
xgb_regressor = XGBRegressor().fit(X_train, Y_train)
```

#### 5.2 Decision Tree Regressor
A simple, interpretable model that often overfits without constraints
```
dt_regressor = DecisionTreeRegressor(random_state=42).fit(X_train, Y_train)
```


### 📊 6. Model Evaluation
Both models were evaluated using standard regression metrics:
- R² Score — variance explained
- MAE — average absolute error
- MSE / RMSE — penalize large errors
- Actual vs Predicted scatter plots

Example:
```
metrics.mean_absolute_error(Y_test, Y_pred)
metrics.r2_score(Y_test, Y_pred)
```
XGBoost generally performed better and generalized more effectively than the Decision Tree.


### ⚙️ 7. Hyperparameter Tuning
#### 7.1 RandomizedSearchCV for XGBoost
Explored combinations of:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`

#### 7.2 GridSearchCV for Decision Tree 
Tuned:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
Cross‑validation (`cv=5`) ensured stable, reliable performance estimates.


### 🧠 8. Interpretation of Results
Key insights:
- XGBoost captured complex relationships and delivered strong performance, especially after tuning.
- Decision Tree was highly flexible but overfit the training data without depth constraints.
- Features like RM, LSTAT, and PTRATIO strongly influenced price predictions.
- Comparing training vs testing metrics highlighted overfitting and guided tuning decisions.

### 🏢 9. Industry Use‑Case Summary
- Real estate agents — setting fair listing prices
- Mortgage lenders — assessing collateral value
- Property developers — identifying undervalued areas
- Insurance companies — evaluating property risk
- Buyers & tenants — understanding what drives housing prices
Accurate price prediction reduces uncertainty and supports smarter investment and lending decisions.


### 📁 10. Project Structure
```
├── data/
│   └── boston_housing_raw.txt
├── notebooks/
│   └── House_Price_Prediction.ipynb
├── README.md
└── LICENSE
```

## 🤝 Contributing
### 🚀 Suggested next steps and improvements
-	Factor notebook logic into testable modules under src/ and add unit tests in tests/.

### 🧭 Style and process
- Tests should import functions from  src/ rather than executing notebook cells..

Thank you for your contributions 🎉

