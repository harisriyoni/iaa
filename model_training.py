import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('iaa_dataset.csv')

# Encode the 'Category' column
category_encoder = LabelEncoder()  # Use a separate encoder for the target
df['Category'] = category_encoder.fit_transform(df['Category'])

# Map 'Gender' to numerical values (e.g., 'L' -> 1, 'P' -> 0)
gender_mapping = {'L': 1, 'P': 0}
df['Gender'] = df['Gender'].map(gender_mapping)

# Drop unnecessary columns
columns_to_drop = ['Date', 'Id']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Split data into features (X) and target (y)
x = df.drop(columns=['Category'])  # Features: 'Age', 'Gender', 'Score'
y = df['Category']  # Target: 'Category'

# Split into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Evaluate the model
y_pred = knn.predict(x_test)
KNN_acc = accuracy_score(y_test, y_pred)
print('Akurasi KNN : {:.2f}%'.format(KNN_acc * 100))
print(classification_report(y_test, y_pred))

# Save the model and encoders
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(category_encoder, 'category_encoder.pkl')

print("Model and encoders have been saved successfully.")
