!pip install textstat

import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import flesch_reading_ease, syllable_count
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset
train_file_path = "/content/lcp_single_train.tsv"
test_file_path = "/content/lcp_single_test.tsv"

# Read the TSV files
train_df = pd.read_csv(train_file_path, sep="\t")
test_df = pd.read_csv(test_file_path, sep="\t")

# Display dataset info
print("Training Dataset Info:")
print(train_df.info())

# Display first five rows
print("\nFirst Five Rows of Training Data:")
print(train_df.head())

# Data Preprocessing
# Checking for missing values
print("\nMissing Values in Dataset:")
print(train_df.isnull().sum())

# Handling missing values
train_df.dropna(inplace=True)

# Handling missing values in X (features)
imputer = SimpleImputer(strategy="mean")
X_train_np = imputer.fit_transform(X_train_np)
X_test_np = imputer.transform(X_test_np)

# Handling missing values in y (target) by dropping rows
y_train_np = y_train_np[~np.isnan(y_train_np)]
y_test_np = y_test_np[~np.isnan(y_test_np)]

print("\nMissing values after imputation:")
print(f"Missing values in X_train: {np.isnan(X_train_np).sum()}")
print(f"Missing values in X_test: {np.isnan(X_test_np).sum()}")
print(f"Missing values in y_train: {np.isnan(y_train_np).sum()}")
print(f"Missing values in y_test: {np.isnan(y_test_np).sum()}")

# Load SpaCy model for text processing
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Apply preprocessing to sentences and tokens
train_df["cleaned_sentence"] = train_df["sentence"].apply(preprocess_text)
train_df["cleaned_token"] = train_df["token"].apply(preprocess_text)

# Tokenization and Lemmatization
train_df["tokenized_sentence"] = train_df["cleaned_sentence"].apply(lambda x: [token.lemma_ for token in nlp(x)])

# Display first five processed rows
print("\nFirst Five Processed Rows:")
print(train_df[["cleaned_sentence", "cleaned_token", "tokenized_sentence" ]].head())

#Exploratory Data Analysis (EDA)
# Complexity Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df["complexity"], bins=30, kde=True, color="blue")
plt.xlabel("Complexity Score")
plt.ylabel("Frequency")
plt.title("Distribution of Complexity Scores")
plt.show()

# Token Length Analysis
train_df["token_length"] = train_df["cleaned_token"].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(train_df["token_length"], bins=20, kde=True, color="green")
plt.xlabel("Token Length (Characters)")
plt.ylabel("Frequency")
plt.title("Distribution of Token Lengths")
plt.show()

# Sentence Length Analysis
train_df["sentence_length"] = train_df["cleaned_sentence"].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 5))
sns.histplot(train_df["sentence_length"], bins=30, kde=True, color="purple")
plt.xlabel("Sentence Length (Words)")
plt.ylabel("Frequency")
plt.title("Distribution of Sentence Lengths")
plt.show()

# Correlation Analysis
plt.figure(figsize=(6, 4))
sns.heatmap(train_df[["complexity", "token_length", "sentence_length"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 4. Feature Extraction
# POS Tagging
train_df["pos_tag"] = train_df["cleaned_token"].apply(lambda x: nlp(x)[0].pos_ if len(nlp(x)) > 0 else "UNKNOWN")

# Readability Scores
train_df["syllable_count"] = train_df["cleaned_token"].apply(syllable_count)
train_df["flesch_reading_ease"] = train_df["cleaned_sentence"].apply(flesch_reading_ease)

# Display first five extracted features
print("\nFirst Five Feature Extraction Results:")
print(train_df[["cleaned_token", "pos_tag", "syllable_count", "flesch_reading_ease"]].head())

# Feature Engineering
# One-Hot Encoding for POS Tags
# Remove 'sparse' argument for older scikit-learn versions
pos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Set sparse=False
pos_encoded = pos_encoder.fit_transform(train_df[["pos_tag"]]) # toarray() is not needed
pos_encoded_df = pd.DataFrame(pos_encoded, columns=pos_encoder.get_feature_names_out(["pos_tag"]))
train_df = pd.concat([train_df, pos_encoded_df], axis=1)

# Scaling Numerical Features
scaler = StandardScaler()
train_df[num_features] = scaler.fit_transform(train_df[num_features])

# Impute numeric features with the mean
imputer = SimpleImputer(strategy="mean")
train_df[num_features] = imputer.fit_transform(train_df[num_features])

# Fill missing values in POS tag columns with 0 (absence of the tag)
pos_tag_columns = [col for col in train_df.columns if col.startswith("pos_tag_")]
train_df[pos_tag_columns] = train_df[pos_tag_columns].fillna(0)

# Drop any remaining rows with missing target values
train_df = train_df.dropna(subset=["complexity"])

print("\nMissing values in the DataFrame after handling:")
print(train_df.isnull().sum())

print("\nChecking for missing values in X and y:")
print(f"Missing values in X_train: {np.isnan(X_train_np).sum()}")
print(f"Missing values in X_test: {np.isnan(X_test_np).sum()}")
print(f"Missing values in y_train: {np.isnan(y_train_np).sum()}")
print(f"Missing values in y_test: {np.isnan(y_test_np).sum()}")

# Display rows with missing values if any
if np.isnan(X_train_np).sum() > 0:
    print("\nRows with missing values in X_train:")
    print(pd.DataFrame(X_train_np).isnull().sum(axis=1))

if np.isnan(y_train_np).sum() > 0:
    print("\nRows with missing values in y_train:")
    print(pd.Series(y_train_np).isnull().sum())

# Checking for missing values in the entire DataFrame
print("\nMissing values in the entire DataFrame before handling:")
print(train_df.isnull().sum())

# Handling missing values in the entire DataFrame
imputer = SimpleImputer(strategy="mean")
train_df[num_features] = imputer.fit_transform(train_df[num_features])

# Drop any remaining rows with missing target values
train_df = train_df.dropna(subset=["complexity"])

print("\nMissing values in the DataFrame after handling:")
print(train_df.isnull().sum())

# 6. Model Definition
# Splitting Data
X = train_df[num_features + list(pos_encoded_df.columns)]
y = train_df["complexity"]
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

# Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
    "Support Vector Regression (SVR)": SVR(kernel='rbf')
}
# Display Model Selection
print("\nModels Defined: Linear Regression, Random Forest, XGBoost, Support Vector Regression (SVR)")

# 7. Training and Evaluation
# Convert to NumPy arrays to avoid dtype issues
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.ravel()
y_test_np = y_test.ravel()

# Training and Evaluation with Regression Metrics
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_np, y_train_np)
    y_pred = model.predict(X_test_np)

    # Regression Metrics
    mse = mean_squared_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)

    print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")

print("\nTraining and Evaluation Complete!")


