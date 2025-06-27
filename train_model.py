import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
from scipy.stats import randint, uniform

# Load dataset
df = pd.read_csv("D:\\diabetes_prediction_system\\cleaned_diabetes_modified.csv")

# Features and target
features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Category']
target = 'Outcome'

X = df[features]
y = df[target]

# Separate numeric and categorical features
numeric_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_features = ['BMI_Category']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)  # drop='first' to avoid dummy variable trap
    ]
)

# Apply preprocessing to entire dataset before SMOTE (SMOTE only for numeric data, so do after encoding)
X_processed = preprocessor.fit_transform(X)

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# Train-test split stratified
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Base models
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Parameter grids for tuning
param_dist_xgb = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
}

param_dist_rf = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
}

param_dist_lr = {
    'C': uniform(0.01, 10),
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("🔍 Tuning XGBClassifier...")
xgb_search = RandomizedSearchCV(
    xgb, param_dist_xgb, n_iter=30, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1, verbose=1
)
xgb_search.fit(X_train, y_train)

print("🔍 Tuning RandomForestClassifier...")
rf_search = RandomizedSearchCV(
    rf, param_dist_rf, n_iter=30, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1, verbose=1
)
rf_search.fit(X_train, y_train)

print("🔍 Tuning LogisticRegression...")
lr_search = RandomizedSearchCV(
    lr, param_dist_lr, n_iter=30, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1, verbose=1
)
lr_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_
best_rf = rf_search.best_estimator_
best_lr = lr_search.best_estimator_

print("Best XGB params:", xgb_search.best_params_)
print("Best RF params:", rf_search.best_params_)
print("Best LR params:", lr_search.best_params_)

# Stacking ensemble
stacked_model = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf),
        ('lr', best_lr)
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=cv,
    n_jobs=-1,
    passthrough=False
)

print("🚀 Training stacked model...")
stacked_model.fit(X_train, y_train)

# Evaluate
y_pred = stacked_model.predict(X_test)
y_proba = stacked_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ ROC AUC: {roc_auc:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and preprocessor (important to apply same transformations on new data)
joblib.dump(stacked_model, "diabetes_model_tuned.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("💾 Tuned model and preprocessor saved successfully!")
