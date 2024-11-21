# EXPANDED ANTIBIOTIC RESISTANCE

### Clone Project
```
git clone https://github.com/ASNProject/expanded_antibiotic_resistance_knn.git
```

### Install Dependensi
```
pip install -r requirements.txt
```

### Jalankan Project
```
python main.py
```
- Note:
  Pilih metode yang ingin digunakan
  ```
  # Using KNN
  # Prediksi Data Baru [['S. pyogenes', 'Levofloxacin', '4.3']]
  result = predict_outcome_knn(bacterial_strain, antibiotic, mic_value)
  # Using Decision Tree
  # result = predict_outcome_decision_tree(bacterial_strain, antibiotic, mic_value)
  # Using Random Forest
  # result = predict_outcome_random_forest(bacterial_strain, antibiotic, mic_value)
  ```

### Screenshot
