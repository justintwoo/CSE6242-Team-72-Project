import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------------------
# Load and clean data
# ----------------------------------------
df = pd.read_csv("realtor-data.zip.csv")
df.drop(columns=["prev_sold_date", "status", "street", "brokered_by"], inplace=True, errors="ignore")
df.dropna(inplace=True)

excluded_states = ['GUAM', 'DISTRICT OF COLUMBIA', 'NEW BRUNSWICK', 'PUERTO RICO', 'VIRGIN ISLANDS', 'PR']
df = df[~df["state"].str.upper().isin(excluded_states)]

df["zip_code"] = df["zip_code"].astype(str)

# ----------------------------------------
# Target encoding
# ----------------------------------------
zip_means = df.groupby("zip_code")["price"].mean()
city_means = df.groupby("city")["price"].mean()
state_means = df.groupby("state")["price"].mean()

df["city_encoded"] = df["city"].map(city_means)
df["state_encoded"] = df["state"].map(state_means)

df.drop(columns=["city", "state"], inplace=True)

# ----------------------------------------
# Outlier removal
# ----------------------------------------
def remove_outliers(df, columns, threshold=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

numeric_cols = ["price", "bed", "bath", "acre_lot", "house_size", "city_encoded", "state_encoded"]
df = remove_outliers(df, numeric_cols)

# ----------------------------------------
# Define features and target
# ----------------------------------------
features = ["bed", "bath", "acre_lot", "house_size", "city_encoded", "state_encoded"]
X = df[features]
y = df["price"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# Baseline CV for all models
# ----------------------------------------
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "Linear Regression": LinearRegression()
}

best_model_name = None
best_cv_score = -np.inf
cv_scores = {}  # Dictionary to store the R² scores

print("Baseline CV R^2 Scores:")
for name, model in models.items():
    score = cross_val_score(model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1).mean()
    cv_scores[name] = score
    print(f"{name}: {score:.4f}")
    if score > best_cv_score:
        best_model_name = name
        best_cv_score = score

print(f"\nBest model: {best_model_name}")

# ----------------------------------------
# Plot R^2 scores for baseline models
# ----------------------------------------
names = list(cv_scores.keys())
scores = list(cv_scores.values())
x = np.arange(len(names))
width = 0.5

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, scores, width, color='lightblue', edgecolor='black')
ax.set_xlabel("Models", fontsize=16)
ax.set_ylabel("R^2 Score", fontsize=16)
ax.set_title("Baseline Model R^2 Scores", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=14)

# Annotate each bar with its respective R^2 score
for i, score in enumerate(scores):
    ax.text(x[i], score + 0.005, f"{score:.2f}", ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.show()

# ----------------------------------------
# Define hyperparameter spaces
# ----------------------------------------
param_grids = {
    "Random Forest": {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    },
    "XGBoost": {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 1.0]
    }
}

# ----------------------------------------
# Tune only the best model
# ----------------------------------------
if best_model_name == "Random Forest":
    base_model = RandomForestRegressor(random_state=42)
elif best_model_name == "XGBoost":
    base_model = XGBRegressor(random_state=42, verbosity=0)
else:
    base_model = LinearRegression()

if best_model_name in param_grids:
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grids[best_model_name],
        n_iter=10,
        cv=3,
        scoring="r^2",
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_scaled, y)
    best_model = search.best_estimator_
    print("\nBest parameters found:", search.best_params_)
else:
    base_model.fit(X_scaled, y)
    best_model = base_model

# ----------------------------------------
# Final evaluation
# ----------------------------------------
y_pred = best_model.predict(X_scaled)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

print(f"\nFinal Model: {best_model_name}")
print(f"Train RMSE: {rmse:.2f}")
print(f"Train MAE: {mae:.2f}")
print(f"Train R^2: {r2:.4f}")
print(f"Train MAPE: {mape:.2f}%")

# ----------------------------------------
# SHAP Explanation (on sample)
# ----------------------------------------
import shap
# Sample 5000 rows for explanation
sample_size = min(5000, X.shape[0])
X_sample = pd.DataFrame(X_scaled, columns=features).sample(sample_size, random_state=42)

# Use TreeExplainer for tree-based models
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)

# Plot summary
print("\nGenerating SHAP summary plot for model interpretation...")
shap.summary_plot(shap_values, X_sample, feature_names=features)
