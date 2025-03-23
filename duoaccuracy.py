import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Duolingo dataset
df = pd.read_csv("E:/Python/SCDIC-1/duo_story_data.csv")

# Convert time column to datetime format
df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")

# Convert time to numerical timestamps
df["timestamp"] = df["time"].astype(int) // 10 ** 9
feature_cols = ["timestamp"]
X = df[feature_cols]
y = df["score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# General Random Forest Model (Before Tuning)
rf_general = RandomForestRegressor(random_state=42)
rf_general.fit(X_train, y_train)
y_pred_rf_general = rf_general.predict(X_test)

# Hyperparameter Tuning
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_tuned = RandomForestRegressor(random_state=42)
rand_search = RandomizedSearchCV(rf_tuned, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42)
rand_search.fit(X_train, y_train)

# Best model
best_rf_reg = rand_search.best_estimator_
y_pred_rf_tuned = best_rf_reg.predict(X_test)

# Save the best model
joblib.dump(best_rf_reg, "enhanced_rf_model.pkl")


# Define performance categories
def categorize_performance(score):
    if score >= 7.5:
        return "High Performance 🚀"
    elif score >= 5:
        return "Moderate Performance 🔄"
    else:
        return "Needs Improvement ⚠️"


# Evaluate models
print("General Random Forest Regression Metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf_general))
print("MSE:", mean_squared_error(y_test, y_pred_rf_general))
print("R2 Score:", r2_score(y_test, y_pred_rf_general))

print("\nTuned Random Forest Regression Metrics:")
print("Best Parameters:", rand_search.best_params_)
print("MAE:", mean_absolute_error(y_test, y_pred_rf_tuned))
print("MSE:", mean_squared_error(y_test, y_pred_rf_tuned))
print("R2 Score:", r2_score(y_test, y_pred_rf_tuned))

# Visualizations
plt.figure(figsize=(8, 5))
sns.histplot(df["score"], bins=10, kde=True)
plt.title("Score Distribution in Duolingo Stories")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred_rf_tuned, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Identity line
plt.title("Actual vs. Predicted Scores")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.show()

df.sort_values(by="time", inplace=True)
plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["score"], marker='o', linestyle='-')
plt.title("Learning Progress Over Time")
plt.xlabel("Date")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

print("Data analysis and prediction complete!")


