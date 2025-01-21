import os
import json

import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from train import train
from test import test


def load_json_data(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filename} tidak ditemukan.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON di file {filename}: {e}")
        return {}


def button_train():
    if os.path.exists('training.png'):
        os.remove('training.png')
        print("Previous 'training.png' deleted.")

    print("Training Start")
    result = train()
    train_result.config(text=result)

    load_image()


def button_test():
    selected_bacteria = bacteria_dropdown.get()
    selected_antibiotic = antibiotic_dropdown.get()
    selected_environment = environment_dropdown.get()

    if not selected_bacteria or not selected_antibiotic or not selected_environment:
        print("Please select all options before testing.")
        return

    high, moderate, low = test(
        bacteria=selected_bacteria,
        antibiotic=selected_antibiotic,
        environment=selected_environment
    )

    high_value.config(text=f"{high[0]}")
    moderate_value.config(text=f"{moderate[0]}")
    low_value.config(text=f"{low[0]}")

    print(f"Testing completed: High={high[0]}, Moderate={moderate[0]}, Low={low[0]}")


def load_image():
    image_path = "training.png"
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            # Resize the image
            image_resized = image.resize((460, 250))
            # Convert the resized image for Tkinter compatibility
            photo = ImageTk.PhotoImage(image_resized)
            # Update the image label
            image_label.config(image=photo)
            image_label.image = photo  # Keep a reference to avoid garbage collection
        except Exception as e:
            image_label.config(text="Training image cannot be loaded.", image='')
            print(f"Error loading image: {e}")
    else:
        image_label.config(text="Training image not found.", image='')


def open_csv():
    # Open the predicted_resistance_levels_mlp.csv file
    file_path = 'predicted_resistance_levels_mlp.csv'

    if os.path.exists(file_path):
        # Load CSV file using pandas
        df = pd.read_csv(file_path)
        print(f"CSV file loaded:\n{df}")

        # Optionally display the CSV data in a new window or use it for further operations
        display_csv_window(df)
    else:
        print("CSV file not found!")


def display_csv_window(df):
    # Create a new window to display the CSV content
    csv_window = tk.Toplevel(root)
    csv_window.title("CSV Data")

    # Create a Text widget to display the data
    text_box = tk.Text(csv_window, wrap=tk.WORD, width=80, height=20)
    text_box.grid(row=0, column=0, padx=10, pady=10)

    # Insert the CSV data into the Text widget
    text_box.insert(tk.END, df.to_string())

    # Make the Text widget read-only
    text_box.config(state=tk.DISABLED)


# # Load JSON data
# data = load_json_data()

# Load file JSON
bacteria_species_data = load_json_data('bacteria_species.json')
antibiotics_data = load_json_data('antibiotics.json')
environments_data = load_json_data('environments.json')

# Konversi ke daftar unik
bacteria_species_options = pd.Series(bacteria_species_data.get('bacteria_species', [])).unique().tolist()
antibiotic_options = pd.Series(antibiotics_data.get('antibiotics', [])).unique().tolist()
environment_options = pd.Series(environments_data.get('environments', [])).unique().tolist()

# Create the main window
root = tk.Tk()
root.title("Resistance Level Prediction")
root.geometry("1024x400")  # Set window size
root.resizable(False, False)

# Create a Label and Dropdown for Bacteria_Species
bacteria_label = tk.Label(root, text="Select Bacteria Species:")
bacteria_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
bacteria_dropdown = ttk.Combobox(root, values=bacteria_species_options, width=16, font=("Arial", 18))
bacteria_dropdown.grid(row=0, column=1, padx=10, pady=10)

# Create a Label and Dropdown for Antibiotic
antibiotic_label = tk.Label(root, text="Select Antibiotic:")
antibiotic_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
antibiotic_dropdown = ttk.Combobox(root, values=antibiotic_options, width=16, font=("Arial", 18))
antibiotic_dropdown.grid(row=1, column=1, padx=10, pady=10)

# Create a Label and Dropdown for Environment
environment_label = tk.Label(root, text="Select Environment:")
environment_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
environment_dropdown = ttk.Combobox(root, values=environment_options, width=16, font=("Arial", 18))
environment_dropdown.grid(row=2, column=1, padx=10, pady=10)

# Create button for train
train_button = tk.Button(root, text="Train", width=12, command=button_train)
train_button.grid(row=0, column=2, rowspan=1, padx=10, pady=10)

train_result = tk.Label(root, text="")
train_result.grid(row=1, column=2, rowspan=1, padx=10, pady=10)

# Label gambar
image_label = tk.Label(root, text="Training image not found.")
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
submit_button = tk.Button(root, text="Start Prediction", command=button_test)
submit_button.grid(row=3, column=0, columnspan=2, pady=10)

# Label Prediction
high_label = tk.Label(root, text="HIGH (%)")
high_label.grid(row=12, column=0, padx=10, pady=10, sticky='ew')
moderate_label = tk.Label(root, text="MODERATE (%)")
moderate_label.grid(row=12, column=1, padx=10, pady=10, sticky='ew')
low_label = tk.Label(root, text="LOW (%)")
low_label.grid(row=12, column=2, padx=10, pady=10, sticky='ew')

# Value Prediction
high_value = tk.Label(root, text="0", font=("Arial", 36))
high_value.grid(row=14, column=0, rowspan=3, padx=20, pady=10, sticky='ew')
moderate_value = tk.Label(root, text="0", font=("Arial", 36))
moderate_value.grid(row=14, column=1, rowspan=3, padx=20, pady=10, sticky='ew')
low_value = tk.Label(root, text="0", font=("Arial", 36))
low_value.grid(row=14, column=2, rowspan=3, padx=20, pady=10, sticky='ew')

# Create button for train
download_label = tk.Label(root, text="Open Prediction CSV")
download_label.grid(row=12, column=3, padx=20, pady=10, sticky='ew')
download_button = tk.Button(root, text="Open Data", width=12, height=2, font=("Arial", 14), command=open_csv)
download_button.grid(row=14, column=3, rowspan=3, padx=10, pady=10)

# Muat gambar saat aplikasi dimulai
load_image()

# Start the Tkinter event loop
root.mainloop()
