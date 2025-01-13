import os

import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Read the CSV data
data_bacteria = pd.read_csv('bacteria.csv')
data_antibiotic = pd.read_csv('antibiotic.csv')
data_environment = pd.read_csv('environment.csv')

# Get the unique values from each column
bacteria_species_options = data_bacteria['Bacteria_Species'].unique().tolist()
antibiotic_options = data_antibiotic['Antibiotic'].unique().tolist()
environment_options = data_environment['Environment'].unique().tolist()

# Create the main window
root = tk.Tk()
root.title("Resistance Level Prediction")
root.geometry("1024x400")  # Set window size
root.resizable(False, False)

# Create a Label and Dropdown for Bacteria_Species
bacteria_label = tk.Label(root, text="Select Bacteria Species:")
bacteria_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
bacteria_dropdown = ttk.Combobox(root, values=bacteria_species_options, width=16)
bacteria_dropdown.grid(row=0, column=1, padx=10, pady=10)

# Create a Label and Dropdown for Antibiotic
antibiotic_label = tk.Label(root, text="Select Antibiotic:")
antibiotic_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
antibiotic_dropdown = ttk.Combobox(root, values=antibiotic_options, width=16)
antibiotic_dropdown.grid(row=1, column=1, padx=10, pady=10)

# Create a Label and Dropdown for Environment
environment_label = tk.Label(root, text="Select Environment:")
environment_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
environment_dropdown = ttk.Combobox(root, values=environment_options, width=16)
environment_dropdown.grid(row=2, column=1, padx=10, pady=10)

# Create button for train
train_button = tk.Button(root, text="Train", width=12)
train_button.grid(row=0, column=2, rowspan=1, padx=10, pady=10)

# Check if the image exists and load it
image_path = "training.png"
if os.path.exists(image_path):
    try:
        image = Image.open(image_path)
        # Resize the image to 200x200
        image_resized = image.resize((460, 250))
        # Convert the resized image for Tkinter compatibility
        photo = ImageTk.PhotoImage(image_resized)
        # Create a Label widget to display the image
        image_label = tk.Label(root, image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
    except Exception as e:
        # Handle cases where the image file exists but cannot be opened
        image_label = tk.Label(root, text="Training image cannot be loaded.")
else:
    # Display a text label if the image is not found
    image_label = tk.Label(root, text="Training image cannot be loaded.")

image_label.grid(row=0, column=3, rowspan=5, padx=10, pady=10)


# Function to show selected values
def show_selections():
    selected_bacteria = bacteria_dropdown.get()
    selected_antibiotic = antibiotic_dropdown.get()
    selected_environment = environment_dropdown.get()

    print(f"Selected Bacteria Species: {selected_bacteria}")
    print(f"Selected Antibiotic: {selected_antibiotic}")
    print(f"Selected Environment: {selected_environment}")


# Add a button to show the selected values
submit_button = tk.Button(root, text="Start Prediction", command=show_selections)
submit_button.grid(row=3, column=0, columnspan=2, pady=10)

# Label Prediction
high_label = tk.Label(root, text="HIGH (%)")
high_label.grid(row=12, column=0, padx=10, pady=10, sticky='ew')
moderate_label = tk.Label(root, text="MODERATE (%)")
moderate_label.grid(row=12, column=1, padx=10, pady=10, sticky='ew')
low_label = tk.Label(root, text="LOW (%)")
low_label.grid(row=12, column=2, padx=10, pady=10, sticky='ew')

# Value Prediction
high_value = tk.Label(root, text="23", font=("Arial", 36))
high_value.grid(row=14, column=0, rowspan=3, padx=20, pady=10, sticky='ew')
moderate_value = tk.Label(root, text="45", font=("Arial", 36))
moderate_value.grid(row=14, column=1, rowspan=3, padx=20, pady=10, sticky='ew')
low_value = tk.Label(root, text="76", font=("Arial", 36))
low_value.grid(row=14, column=2, rowspan=3, padx=20, pady=10, sticky='ew')

# Create button for train
download_label = tk.Label(root, text="Download Prediction CSV")
download_label.grid(row=12, column=3, padx=20, pady=10, sticky='ew')
download_button = tk.Button(root, text="Download Data", width=12, height=2, font=("Arial", 14))
download_button.grid(row=14, column=3, rowspan=3, padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()
