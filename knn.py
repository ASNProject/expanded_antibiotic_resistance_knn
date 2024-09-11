import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Membaca data csv
data = pd.read_csv('data.csv')

# Preprocessing
label_encoder = {
    'Bacterial Strain': LabelEncoder(),
    'Antibiotic': LabelEncoder(),
}

for col, encoder in label_encoder.items():
    data[col] = encoder.fit_transform(data[col])

data['MIC (μg/mL)'] = pd.to_numeric(data['MIC (μg/mL)'], errors='coerce')

X = data[['Bacterial Strain', 'Antibiotic', 'MIC (μg/mL)']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


def predict_outcome_knn(bacterial_strain, antibiotic, mic):
    new_data = [[bacterial_strain, antibiotic, mic]]

    new_data_encoded = [
        label_encoder['Bacterial Strain'].transform([new_data[0][0]])[0],
        label_encoder['Antibiotic'].transform([new_data[0][1]])[0],
        float(new_data[0][2]),
    ]

    # Mengubah data baru menjadi array 2D
    # new_data_encoded = [new_data_encoded]

    new_data_scaled = scaler.transform([new_data_encoded])
    prediction = knn.predict(new_data_scaled)
    return prediction[0]
