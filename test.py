import os

import pandas as pd
import joblib
from datetime import datetime


def test(bacteria, antibiotic, environment):
    # 1. Load the trained model and label encoders
    model = joblib.load('trained_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')  # This should be the encoder saved during training
    label_encoder_target = joblib.load('label_encoder_target.pkl')

    # 2. Define the input data as variables
    # bacteria = 'S.aureus'
    # antibiotic = 'Vancomycin'
    # environment = 'Hospital'

    # 3. Prepare the test data using the variables
    test_data = pd.DataFrame([
        {'Bacteria_Species': bacteria, 'Antibiotic': antibiotic, 'Environment': environment}
    ])

    # 4. Preprocess the test data (transform using the same encoder)
    def safe_transform(label_encoder, label):
        try:
            return label_encoder.transform([label])[0]
        except ValueError:
            # Handle the unseen label case (for example, you can return a default value or skip it)
            return -1  # Or handle it differently based on your use case

    test_data['Bacteria_Species'] = test_data['Bacteria_Species'].apply(lambda x: safe_transform(label_encoder, x))
    test_data['Antibiotic'] = test_data['Antibiotic'].apply(lambda x: safe_transform(label_encoder, x))
    test_data['Environment'] = test_data['Environment'].apply(lambda x: safe_transform(label_encoder, x))

    # 5. Make predictions
    X_test_data = test_data[['Bacteria_Species', 'Antibiotic', 'Environment']]
    predictions = model.predict(X_test_data)

    # 6. Calculate probabilities
    probabilities = model.predict_proba(X_test_data)

    # 7. Calculate percentage probabilities for each class
    high_prob = (probabilities[:, 0] * 100).round(0).astype(int)
    moderate_prob = (probabilities[:, 1] * 100).round(0).astype(int)
    low_prob = (probabilities[:, 2] * 100).round(0).astype(int)

    # 8. Reverse encoding for predictions
    # y_pred_actual = label_encoder_target.inverse_transform(predictions)

    # 9. Combine the results into a DataFrame
    output = pd.DataFrame({
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * len(high_prob),  # Current timestamp for each row
        'High%': high_prob,
        'Moderate%': moderate_prob,
        'Low%': low_prob
    })

    # 10. Check if CSV file exists
    file_path = 'predicted_resistance_levels_mlp.csv'
    if os.path.exists(file_path):
        # If file exists, load it and append new data
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, output], ignore_index=True)  # Pass as a list
        updated_data.to_csv(file_path, index=False)  # Save updated data back to CSV
    else:
        # If file doesn't exist, create a new one with the output
        output.to_csv(file_path, index=False)

    # 11. Print the results
    print(output)

    return high_prob, moderate_prob, low_prob
