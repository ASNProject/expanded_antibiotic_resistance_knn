import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib


def train():
    # 1. Load dataset
    data = pd.read_csv('data.csv')

    # 2. Preprocessing (encoding)
    label_encoder = LabelEncoder()
    data['Bacteria_Species'] = label_encoder.fit_transform(data['Bacteria_Species'])
    data['Antibiotic'] = label_encoder.fit_transform(data['Antibiotic'])
    data['Environment'] = label_encoder.fit_transform(data['Environment'])

    label_encoder_target = LabelEncoder()
    data['Resistance_Level'] = label_encoder_target.fit_transform(data['Resistance_Level'])

    X = data[['Bacteria_Species', 'Antibiotic', 'Environment']]
    y = data['Resistance_Level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the MLP model and capture the loss history
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, solver='adam', warm_start=True)

    # Fit the model and capture loss during each iteration
    train_losses = []
    for _ in range(model.max_iter):
        model.fit(X_train, y_train)
        train_losses.append(model.loss_)

    # 4. Plot the training loss
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss During MLP Training')
    plt.legend()
    plt.savefig('training.png', format='png')

    # 5. Save the trained model and label encoders
    joblib.dump(model, 'trained_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the label encoder for later use
    joblib.dump(label_encoder_target, 'label_encoder_target.pkl')  # Save the target label encoder
