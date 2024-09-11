import tkinter as tk
from knn import predict_outcome_knn
from decision_tree import predict_outcome_decision_tree
from random_forest import predict_outcome_random_forest


# Fungsi untuk memproses input dan melakukan prediksi
def run_prediction():
    bacterial_strain = strain_entry.get()
    antibiotic = antibiotic_entry.get()
    mic_value = mic_entry.get()

    try:
        mic_value = float(mic_value)
        # Using KNN
        # Prediksi Data Baru [['S. pyogenes', 'Levofloxacin', '4.3']]
        result = predict_outcome_knn(bacterial_strain, antibiotic, mic_value)
        # Using Decision Tree
        # result = predict_outcome_decision_tree(bacterial_strain, antibiotic, mic_value)
        # Using Random Forest
        # result = predict_outcome_random_forest(bacterial_strain, antibiotic, mic_value)
        result_label.config(text=f"Predicted Outcome: {result}")
    except ValueError:
        result_label.config(text="Error: MIC value harus berupa angka.")


# Membuat jendela utama
root = tk.Tk()
root.title("Expanded Antibiotic Resistance")

# Label dan Entry untuk Bacterial Strain
strain_label = tk.Label(root, text="Bacterial Strain:")
strain_label.grid(row=0, column=0, padx=10, pady=10)
strain_entry = tk.Entry(root)
strain_entry.grid(row=0, column=1, padx=10, pady=10)

# Label dan Entry untuk Antibiotic
antibiotic_label = tk.Label(root, text="Antibiotic:")
antibiotic_label.grid(row=1, column=0, padx=10, pady=10)
antibiotic_entry = tk.Entry(root)
antibiotic_entry.grid(row=1, column=1, padx=10, pady=10)

# Label dan Entry untuk MIC (μg/mL)
mic_label = tk.Label(root, text="MIC (μg/mL):")
mic_label.grid(row=2, column=0, padx=10, pady=10)
mic_entry = tk.Entry(root)
mic_entry.grid(row=2, column=1, padx=10, pady=10)

# Tombol untuk menjalankan prediksi
predict_button = tk.Button(root, text="Predict", command=run_prediction)
predict_button.grid(row=3, column=0, columnspan=2, pady=20)

# Label untuk menampilkan hasil prediksi
result_label = tk.Label(root, text="")
result_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# Menjalankan aplikasi
root.mainloop()
