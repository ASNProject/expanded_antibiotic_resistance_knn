import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Load the training data
df_train = pd.read_csv('data2.csv')

# Label Encoding the categorical columns for training data
le_bacteria = LabelEncoder()
le_antibiotic = LabelEncoder()
le_env = LabelEncoder()
le_response = LabelEncoder()

df_train['Bacteria_Species'] = le_bacteria.fit_transform(df_train['Bacteria_Species'])
df_train['Antibiotic'] = le_antibiotic.fit_transform(df_train['Antibiotic'])
df_train['Environment'] = le_env.fit_transform(df_train['Environment'])
df_train['Resistance_Level'] = le_response.fit_transform(df_train['Resistance_Level'])

# Define features and target for training data
X_train = df_train[['Bacteria_Species', 'Antibiotic', 'Minimum_Inhibitory_Concentration (MIC)', 'Mutation_Frequency']]  # Features
y_train = df_train['Resistance_Level']  # Target variable

# Normalize features for training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# One-hot encode the target variable for training data
y_train_encoded = to_categorical(y_train)

# Build the deep learning model (ANN)
model = Sequential()

# Input Layer
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))

# Hidden Layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Output Layer
model.add(Dense(y_train_encoded.shape[1], activation='softmax'))  # Softmax for multi-class classification

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_encoded, epochs=500, batch_size=32, verbose=1)

# Load the test data
df_test = pd.read_csv('test2.csv')

# Label Encoding the categorical columns for test data
df_test['Bacteria_Species'] = le_bacteria.transform(df_test['Bacteria_Species'])
df_test['Antibiotic'] = le_antibiotic.transform(df_test['Antibiotic'])
df_test['Environment'] = le_env.transform(df_test['Environment'])
df_test['Resistance_Level'] = le_response.transform(df_test['Resistance_Level'])

# Define features and target for test data
X_test = df_test[['Bacteria_Species', 'Antibiotic', 'Minimum_Inhibitory_Concentration (MIC)', 'Mutation_Frequency']]  # Features
y_test = df_test['Resistance_Level']  # Target variable

# Normalize features for test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# One-hot encode the target variable for test data
y_test_encoded = to_categorical(y_test)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_scaled, y_test_encoded)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)
y_pred_classes = y_pred.argmax(axis=-1)  # Get the class with the highest probability

# Convert the predictions back to the original labels
y_pred_labels = le_response.inverse_transform(y_pred_classes)

# Convert the true labels from one-hot encoding to integer labels for evaluation
y_true_labels = le_response.inverse_transform(y_test_encoded.argmax(axis=-1))

# Print results
print("Predicted Resistance Levels:")
print(y_pred_labels)

# Print the actual true labels for comparison
print("Actual Resistance Levels:")
print(y_true_labels)

# Optionally, you can evaluate the accuracy on the raw class labels
test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"Test Accuracy (with raw labels): {test_accuracy * 100:.2f}%")
