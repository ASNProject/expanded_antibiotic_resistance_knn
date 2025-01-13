from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
data = pd.read_csv('data.csv')

input_data = [
    # {'Bacteria_Species': 'E.coli', 'Antibiotic': 'Ampicillin', 'Environment': 'Hospital', 'Resistance_Level': 'High'},
    {'Bacteria_Species': 'S.aureus', 'Antibiotic': 'Vancomycin', 'Environment': 'Hospital', 'Resistance_Level': 'Moderate'},
    # {'Bacteria_Species': 'P.aeruginosa', 'Antibiotic': 'Ciprofloxacin', 'Environment': 'Hospital',
    # 'Resistance_Level': 'Low'}
]

# 2. Buat DataFrame dari input variabel
test_data = pd.DataFrame(input_data)
# test_data = pd.read_csv('../test.csv')

# 2. Combine the training and test datasets for encoding purposes
combined_data = pd.concat([data[['Bacteria_Species', 'Antibiotic', 'Environment']], test_data[['Bacteria_Species', 'Antibiotic', 'Environment']]])

# 3. Initialize and fit the LabelEncoder on the combined data
label_encoder = LabelEncoder()
combined_data['Bacteria_Species'] = label_encoder.fit_transform(combined_data['Bacteria_Species'])
combined_data['Antibiotic'] = label_encoder.fit_transform(combined_data['Antibiotic'])
combined_data['Environment'] = label_encoder.fit_transform(combined_data['Environment'])

# 4. Separate the combined data back into the training and test datasets
data[['Bacteria_Species', 'Antibiotic', 'Environment']] = combined_data[['Bacteria_Species', 'Antibiotic', 'Environment']].iloc[:len(data)]
test_data[['Bacteria_Species', 'Antibiotic', 'Environment']] = combined_data[['Bacteria_Species', 'Antibiotic', 'Environment']].iloc[len(data):]

# 5. Encode target variable (Resistance_Level) only in training data
label_encoder_target = LabelEncoder()
data['Resistance_Level'] = label_encoder_target.fit_transform(data['Resistance_Level'])
test_data['Resistance_Level'] = label_encoder_target.transform(test_data['Resistance_Level'])  # Test data must match training labels

# 6. Split the features (X) and target variable (y)
X = data[['Bacteria_Species', 'Antibiotic', 'Environment']]  # Only select relevant features
y = data['Resistance_Level']

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the MLP model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, solver='adam')
model.fit(X_train, y_train)

# 9. Make predictions on the test data (only relevant features)
X_test_data = test_data[['Bacteria_Species', 'Antibiotic', 'Environment']]  # Select only relevant features
predictions = model.predict(X_test_data)

# 10. Calculate probabilities for each class
probabilities = model.predict_proba(X_test_data)

# 11. Calculate percentage probabilities for each class
high_prob = (probabilities[:, 0] * 100).round(1)  # Class 'High'
moderate_prob = (probabilities[:, 1] * 100).round(1)  # Class 'Moderate'
low_prob = (probabilities[:, 2] * 100).round(1)  # Class 'Low'

# 12. Reverse the encoding for actual and predicted labels
y_test_actual = label_encoder_target.inverse_transform(test_data['Resistance_Level'])
y_pred_actual = label_encoder_target.inverse_transform(predictions)

# 13. Combine the actual results and predicted results into a new DataFrame
output = pd.DataFrame({
    'Actual Resistance Level': y_test_actual,
    'Predicted Resistance Level': y_pred_actual,
    'High%': high_prob,
    'Moderate%': moderate_prob,
    'Low%': low_prob
})

# 14. Save the first output to a CSV file
output.to_csv('predicted_resistance_levels_mlp.csv', index=False)

# 15. Print the first output for review
print(output)

# 16. Evaluate the model on the split test set
print("Hasilnya")
y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

# 17. Create a second output CSV for the input features and percentage probabilities
input_output = pd.DataFrame({
    'Bacteria_Species': test_data['Bacteria_Species'],
    'Antibiotic': test_data['Antibiotic'],
    'Environment': test_data['Environment'],
    'High%': high_prob,
    'Moderate%': moderate_prob,
    'Low%': low_prob
})

# 18. Save the second output to a CSV file
input_output.to_csv('input_with_probabilities_mlp.csv', index=False)

# 19. Print the second output for review
print(input_output)

