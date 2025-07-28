import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.preprocessing import preprocess_data
import config

# Load dataset
df = pd.read_csv(config.DATA_PATH)

# Basic preprocessing
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X = preprocess_data(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save pipeline
joblib.dump(clf, config.MODEL_PATH)
