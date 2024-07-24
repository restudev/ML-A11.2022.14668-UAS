#### Nama     : Restu Lestari Mulianingrum

#### NIM      : A11.2022.14668

#### Kelompok : A11.4413
# **PREDIKSI CHURN DI PERUSAHAAN TELEKOMUNIKASI DENGAN METODE XGBOOST**
## **Ringkasan Project**
Proyek ini berfokus pada analisis churn pelanggan di industri telekomunikasi. Churn pelanggan adalah fenomena di mana pelanggan berhenti menggunakan layanan dari penyedia layanan tertentu. Mengidentifikasi faktor-faktor yang menyebabkan churn dan mengembangkan model prediksi dapat membantu perusahaan telekomunikasi mengurangi tingkat churn dan meningkatkan retensi pelanggan.
## **Permasalahan**
1. **Identifikasi Faktor Penyebab Churn**: Menentukan variabel-variabel utama yang berkontribusi terhadap churn pelanggan.
2. **Prediksi Churn**: Mengembangkan model yang dapat memprediksi apakah seorang pelanggan akan churn atau tidak berdasarkan data historis.
3. **Intervensi yang Efektif**: Menemukan strategi intervensi yang efektif untuk mengurangi churn berdasarkan hasil analisis dan prediksi.

## **Tujuan yang Akan Dicapai**

1. **Analisis Deskriptif**: Menganalisis data pelanggan untuk memahami karakteristik umum dan pola-pola yang berkaitan dengan churn.
2. **Pengembangan Model Prediksi**: Membangun model machine learning untuk memprediksi churn pelanggan.
3. **Rekomendasi Strategi**: Memberikan rekomendasi strategis berdasarkan hasil analisis dan model prediksi untuk mengurangi churn pelanggan.

## **Model / Alur Penyelesaian**

1. **Pra-pemrosesan Data**: Membersihkan dan mempersiapkan data untuk analisis lebih lanjut, termasuk penanganan data yang hilang dan normalisasi.
2. **Analisis Eksploratif**: Melakukan analisis eksploratif untuk memahami karakteristik data dan mengidentifikasi pola-pola yang signifikan.
3. **Pengembangan Model**:
   - **Pemilihan Model**: Memilih algoritma machine learning yang sesuai. Untuk proyek ini, kita akan menggunakan **XGBoost (Extreme Gradient Boosting)**.
   - **Pelatihan Model**: Melatih model menggunakan data pelatihan dan melakukan validasi silang untuk memastikan kinerja yang baik.
   - **Evaluasi Model**: Mengevaluasi model menggunakan metrik seperti akurasi, precision, recall, dan F1-score.
4. **Prediksi dan Interpretasi (Output)**: Menggunakan model untuk memprediksi churn pada data baru dan menginterpretasikan hasil prediksi untuk mengidentifikasi faktor-faktor risiko.

![Deskripsi Gambar](images/bagan.png)

## **Memuat Data dan Mengimpor Modul**

# Mengimpor paket-paket yang diperlukan
import pandas as pd
import numpy as np
# Mengimpor paket visualisasi
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# membaca data

telecom_df = pd.read_csv('Telecom_Churn.csv')

print('Data berhasil terbaca!')
telecom_df.head()
telecom_df.info()
# **Memahami Lebih Lanjut Data**
# Melihat data dari 5 baris teratas untuk mendapatkan gambaran tentang data
telecom_df.head(5)
# Melihat data dari 5 baris terbawah untuk mendapatkan gambaran tentang data
telecom_df.tail(5)
# Mendapatkan bentuk dataset dengan jumlah baris dan kolom
print(telecom_df.shape)
# Mendapatkan semua kolom
print("Features of the dataset:")
telecom_df.columns
### **Rincian Fitur:**

**STATE:**
51 Kode Negara

**Account Length:**
Durasi Akun

**Area Code:**
Kode Nomor Area yang mencakup beberapa Negara Bagian

**International Plan:**
"Yes" menunjukkan ada Langganan Internasional dan "No" menunjukkan tidak ada langganan untuk Rencana Internasional

**Voice Mail Plan:**
"Yes" menunjukkan ada Rencana Suara dan "No" menunjukkan tidak ada langganan untuk Rencana Suara

**Number vmail messages:**
Jumlah Pesan Suara yang berkisar dari 0 hingga 50

**Total day minutes:**
Jumlah Total Menit yang Dihabiskan di Pagi Hari

**Total day calls:**
Jumlah Total Panggilan yang Dilakukan di Pagi Hari

**Total day charge:**
Total Biaya yang Dibebankan kepada Pelanggan di Pagi Hari

**Total eve minutes:**
Jumlah Total Menit yang Dihabiskan di Malam Hari

**Total eve calls:**
Jumlah Total Panggilan yang Dilakukan di Malam Hari

**Total eve charge:**
Total Biaya yang Dibebankan kepada Pelanggan di Malam Hari

**Total night minutes:**
Jumlah Total Menit yang Dihabiskan di Malam Hari

**Total night calls:**
Jumlah Total Panggilan yang Dilakukan di Malam Hari

**Total night charge:**
Total Biaya yang Dibebankan kepada Pelanggan di Malam Hari

**Customer service calls:**
Jumlah Panggilan Layanan Pelanggan yang Dilakukan oleh Pelanggan

**Churn:**
Churn Pelanggan, "True" berarti pelanggan yang churn, "False" berarti pelanggan yang tetap
