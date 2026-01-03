NAMA      : RIZKY ZEHANS ONASSIS
NIM       : 231011400496
MATKUL    : PENGOLAHAN CITRA DIGITAL

🐟 Sistem Klasifikasi Jenis Ikan Cupang (Betta Fish Classification)

Proyek ini adalah sistem otomatis untuk mengidentifikasi jenis ikan cupang menggunakan teknik **Pengolahan Citra Digital** dan algoritma **Machine Learning**. Sistem ini mengekstraksi fitur visual dari gambar dan mengklasifikasikannya ke dalam beberapa jenis (Crowntail, Doubletail, Halfmoon, Plakat).

---

🚀 Fitur Utama
* **Segmentasi Otomatis**: Memisahkan objek ikan dari latar belakang menggunakan *Otsu Thresholding*.
* **Ekstraksi Fitur Multi-Dimensi**:
    * **Warna**: Mean & Std Dev pada ruang warna BGR dan HSV.
    * **Bentuk**: Area, Perimeter, Solidity, dan Hu Moments.
    * **Tekstur**: Analisis pola menggunakan GLCM dan LBP.
* **Klasifikasi Pintar**: Menggunakan algoritma *Random Forest Classifier* untuk akurasi yang stabil.
* **Antarmuka GUI**: Dibangun dengan Python Tkinter untuk kemudahan penggunaan.

---

🛠️ Teknologi & Library
Proyek ini menggunakan bahasa pemrograman **Python 3** dengan library utama:
* `OpenCV`: Pemrosesan gambar dan deteksi kontur.
* `Scikit-Image`: Ekstraksi fitur tekstur dan morfologi.
* `Scikit-Learn`: Algoritma klasifikasi Machine Learning.
* `Pandas & Numpy`: Manajemen data dataset fitur (CSV).
* `Tkinter & Pillow`: Pembuatan antarmuka grafis (GUI).

---

📂 Struktur File
* `dataset/`: Folder berisi sub-folder foto ikan berdasarkan kelasnya (misal: `/crowntail`, `/plakat`).
* `fiturcupang.py`: Script untuk ekstraksi fitur masal dan pembuatan dataset CSV.
* `hasil_fitur.csv`: Database hasil ekstraksi fitur visual.
* `identifikasi_cupang_v2.py`: Aplikasi utama (GUI) dan proses klasifikasi AI.

---

📝 Cara Penggunaan

1. Ekstraksi Fitur (Preprocessing)
Jalankan script berikut untuk membuat database fitur dari folder dataset:
```bash
python fiturcupang.py dataset/ hasil_fitur.csv

2. Menjalankan Aplikasi
Setelah file hasil_fitur.csv terbentuk, jalankan aplikasi GUI:

"python identifikasi_cupang_v2.py"

Klik tombol "Pilih Gambar Ikan" untuk mulai mengidentifikasi.

👥 Anggota Kelompok (UAS Citra Digital)
1. RIZKY ZEHANS ONASSIS (231011400496)

2. IRGI FEBRIAN (231011401642)

3. REI MAKTAL ALFARIDZI (231011400507)

4. MIKHAEL KELLY JACONIA SAPTENNO (231011402248)

5. SATRIA RAMADHAN (231011401641)

Jalankan script berikut untuk membuat database fitur dari folder dataset:
```bash
python fiturcupang.py dataset/ hasil_fitur.csv
