import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Membaca dataset
data = pd.read_csv('data.csv')

# Convert to pandas dataframe
df = pd.DataFrame(data)

# Encode categorical features
label_encoder_bacterial = LabelEncoder()
label_encoder_antibiotic = LabelEncoder()
label_encoder_outcome = LabelEncoder()

df['bacterial_strain_encoded'] = label_encoder_bacterial.fit_transform(df['Bacterial Strain'])
df['antibiotic_encoded'] = label_encoder_antibiotic.fit_transform(df['Antibiotic'])
df['outcome_encoded'] = label_encoder_outcome.fit_transform(df['Outcome'])

# Pisahkan fitur (X) dan target (y)
X = df[['bacterial_strain_encoded', 'antibiotic_encoded', 'MIC (μg/mL)']]
y = df['outcome_encoded']

# Pisahkan data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Inisialisasi model Decision Tree
model = DecisionTreeClassifier()

# Latih model menggunakan data training
model.fit(X_train, y_train)


# Fungsi prediksi data baru
def predict_outcome_decision_tree(bacterial_strain, antibiotic, mic):
    # Encode new data
    bacterial_strain_encoded = label_encoder_bacterial.transform([bacterial_strain])[0]
    antibiotic_encoded = label_encoder_antibiotic.transform([antibiotic])[0]

    # Prediksi
    prediksi_baru = model.predict([[bacterial_strain_encoded, antibiotic_encoded, mic]])
    prediksi_baru_decoded = label_encoder_outcome.inverse_transform(prediksi_baru)

    return prediksi_baru_decoded[0]
