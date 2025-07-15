import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("student-mat.csv", sep=';')  # replace with your dataset path

# Preprocessing
le = LabelEncoder()
for col in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'activities']:
    df[col] = le.fit_transform(df[col])

# Target column: G3 >= 10 -> pass
df['passed'] = (df['G3'] >= 10).astype(int)
features = df.drop(columns=['G1', 'G2', 'G3', 'passed'])
target = df['passed']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Streamlit app
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# Sidebar
st.sidebar.header("Project Details")
st.sidebar.write("Predict student performance based on academic and personal data.")
st.sidebar.write("Model: Random Forest Classifier")

# Data preview
with st.expander("📊 View Raw Data"):
    st.dataframe(df.head())

# Prediction outcome
st.subheader("📈 Prediction Results")
st.text("Classification Report")
st.code(classification_report(y_test, y_pred), language='text')

# Confusion matrix plot
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Feature importance
st.subheader("📌 Feature Importances")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
important_features = features.columns[indices]
importance_values = importances[indices]

fig2, ax2 = plt.subplots()
sns.barplot(x=importance_values, y=important_features, ax=ax2)
plt.title("Top Predictive Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(fig2)

st.success("Dashboard is live and interactive! Upload more student data for future predictions.")
