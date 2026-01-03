import pandas as pd
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier

# Mengambil fungsi ekstraksi dari file fiturcupang.py
try:
    from fiturcupang import extract_features_for_image
except ImportError:
    print("Error: File 'fiturcupang.py' harus ada di folder yang sama!")

class SistemIdentifikasi:
    def __init__(self, window):
        self.window = window
        self.window.title("Sistem Klasifikasi Jenis Ikan Cupang")
        self.window.geometry("500x550")

        # 1. Melatih Sistem berdasarkan database
        self.latih_sistem()

        # 2. Elemen Antarmuka (UI)
        self.label_judul = Label(window, text="Sistem Klasifikasi Ikan Cupang", font=("Arial", 16, "bold"))
        self.label_judul.pack(pady=10)

        self.canvas = tk.Canvas(window, width=300, height=300, bg="gray")
        self.canvas.pack(pady=10)

        self.btn_pilih = Button(window, text="Pilih Gambar Ikan", command=self.pilih_dan_proses, 
                                bg="#2196F3", fg="white", font=("Arial", 12), padx=20)
        self.btn_pilih.pack(pady=10)

        self.label_hasil = Label(window, text="Hasil Identifikasi: -", font=("Arial", 14))
        self.label_hasil.pack(pady=10)

    def latih_sistem(self):
        csv_file = "hasil_fitur.csv"
        if not os.path.exists(csv_file):
            messagebox.showerror("Error", "Database 'hasil_fitur.csv' tidak ditemukan!")
            self.window.destroy()
            return

        data = pd.read_csv(csv_file)
        X = data.drop(['filename', 'label'], axis=1)
        y = data['label']

        # Menggunakan algoritma Random Forest untuk melatih sistem
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.X_columns = X.columns 

    def pilih_dan_proses(self):
        path = filedialog.askopenfilename()
        if path:
            # Menampilkan gambar pada sistem
            img_display = Image.open(path)
            img_display = img_display.resize((300, 300), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_display)
            self.canvas.create_image(150, 150, image=img_tk)
            self.canvas.image = img_tk

            try:
                # Proses Ekstraksi Fitur Visual oleh Sistem
                fitur_gambar = extract_features_for_image(path)
                df_baru = pd.DataFrame([fitur_gambar])
                X_input = df_baru.drop(['filename', 'label'], axis=1)
                X_input = X_input[self.X_columns]

                # Sistem melakukan prediksi jenis ikan
                hasil = self.model.predict(X_input)[0]

                # Menampilkan hasil identifikasi sistem
                self.label_hasil.config(text=f"Hasil Identifikasi: {hasil.upper()}", fg="#0D47A1")

            except Exception as e:
                messagebox.showerror("Error", f"Sistem gagal memproses data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SistemIdentifikasi(root)
    root.mainloop()