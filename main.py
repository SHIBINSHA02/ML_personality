import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
df = pd.read_csv('data.csv')

# Inspect the data for missing values
print("Missing values before preprocessing:")
print(df.isna().sum())

# Handle missing values
# Numeric columns: Impute with median
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: Impute with mode
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Verify no missing values remain
print("\nMissing values after preprocessing:")
print(df.isna().sum())

# Encode categorical variables
le = LabelEncoder()
df['Stage_fear'] = le.fit_transform(df['Stage_fear'])  # Yes=1, No=0
df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])  # Yes=1, No=0
df['Personality'] = le.fit_transform(df['Personality'])  # Extrovert=1, Introvert=0

# Define features (X) and target (y)
X = df[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 
        'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']]
y = df['Personality']

# Check for NaN in X
print("\nChecking for NaN in features (X):")
print(X.isna().sum())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print model performance
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")