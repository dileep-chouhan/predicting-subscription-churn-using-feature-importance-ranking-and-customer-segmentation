import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
n_samples = 500
data = {
    'tenure': np.random.randint(1, 25, n_samples),
    'monthly_bill': np.random.uniform(20, 100, n_samples),
    'customer_service_calls': np.random.randint(0, 5, n_samples),
    'internet_speed': np.random.choice(['High', 'Medium', 'Low'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 30% churn rate
}
df = pd.DataFrame(data)
# Convert categorical feature to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['internet_speed'], drop_first=True)
# --- 2. Data Preprocessing and Feature Engineering ---
X = df.drop('churn', axis=1)
y = df['churn']
# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=2) # Reduce to 2 principal components for visualization
X_pca = pca.fit_transform(X_scaled)
# --- 3. Model Training and Feature Importance ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
# --- 4. Visualization ---
# Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")
# PCA Visualization (if PCA was applied)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=['blue', 'red'])
plt.title('Customer Segmentation using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('customer_segmentation.png')
print("Plot saved to customer_segmentation.png")
plt.show()