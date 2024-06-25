import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Memuat kumpulan data
data = pd.read_csv('ai4i2020.csv')

# Menampilkan beberapa baris pertama kumpulan data
print(data.head())

data = data.drop(["UDI","Product ID","Type","TWF","HDF","PWF","OSF","RNF"], axis=1)

# Fitur dan target terpisah
X = data.drop('Machine failure', axis=1)  # Fitur
y = data['Machine failure']  # variabel target

# Menyandikan variabel kategori jika perlu
X = pd.get_dummies(X)

# Menangani nilai yang hilang jika perlu
X = X.fillna(X.mean())

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Membuat prediksi
y_pred = model.predict(X_test)

# evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

plt.scatter(range(len(y_pred)), y_pred)
plt.show()