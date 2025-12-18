import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ✅ Load your dataset (from Excel file)
df = pd.read_excel("C:/Users/nnaya/Desktop/math chart boat/large_math_question_dataset.xlsx")

# ✅ Split data into features and labels
X = df["question"]
y = df["label"]

# ✅ Convert text data into numerical vectors
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# ✅ Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# ✅ Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ✅ Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ✅ Save the model and vectorizer
joblib.dump(model, "math_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
