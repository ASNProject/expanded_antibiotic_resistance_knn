import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Membaca data csv
data = pd.read_csv('data.csv')

# Preprocessing
# Membuat LabelEncoder untuk masing-masing kolom kategorikal
label_encoder_strain = LabelEncoder()
label_encoder_antibiotic = LabelEncoder()
label_encoder_mic = LabelEncoder()

# Mengubah kolom menjadi numerik
data['Bacterial Strain'] = label_encoder_strain.fit_transform(data['Bacterial Strain'])
data['Antibiotic'] = label_encoder_antibiotic.fit_transform(data['Antibiotic'])
data['MIC (μg/mL)'] = label_encoder_mic.fit_transform(data['MIC (μg/mL)'])

# Fitur (X) dan target (y)
X = data[['Bacterial Strain', 'Antibiotic', 'MIC (μg/mL)']]
y = data['Outcome']

# Membagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat model Random Forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)


# Fungsi untuk memprediksi data baru
def predict_outcome_random_forest(bacterial_strain, antibiotic, mic_value):
    # Mengubah data baru menjadi bentuk numerik menggunakan encoder yang sudah dilatih
    bacterial_strain_encoded = label_encoder_strain.transform([bacterial_strain])[0]
    antibiotic_encoded = label_encoder_antibiotic.transform([antibiotic])[0]

    # Gabungkan data menjadi satu array
    new_data_encoded = [[bacterial_strain_encoded, antibiotic_encoded, float(mic_value)]]

    # Skalakan data baru
    new_data_scaled = scaler.transform(new_data_encoded)

    # Prediksi hasil menggunakan model
    prediction = random_forest.predict(new_data_scaled)

    return prediction[0]
