# Optimasi Model LSTM & GRU dengan Hyperparameter Tuning untuk Prediksi Harga Saham

Tugas ini merupakan tugas lanjutan untuk meningkatkan akurasi model *deep learning* dalam memprediksi harga saham **Samsung (005930.KS)**. Fokus utama dari analisis ini adalah melakukan *hyperparameter tuning* secara otomatis pada arsitektur **LSTM (Long Short-Term Memory)** dan **GRU (Gated Recurrent Unit)** untuk menemukan kombinasi parameter terbaik yang menghasilkan performa prediksi paling optimal.

## Latar Belakang

Prediksi harga saham adalah tantangan kompleks karena sifat data yang *time-series* dan dinamis. Meskipun model seperti LSTM dan GRU sangat efektif dalam menangkap pola sekuensial, performanya sangat bergantung pada pemilihan hyperparameter yang tepat, seperti jumlah unit, *dropout rate*, dan *learning rate*. Proses tuning manual seringkali memakan waktu dan tidak efisien. Oleh karena itu, proyek ini memanfaatkan *library* **Keras Tuner** untuk mengotomatisasi pencarian ini.

## Dataset

Data yang digunakan adalah data historis harga saham **Samsung (005930.KS)** yang diunduh menggunakan *library* `yfinance`. Periode data yang diambil adalah dari **31 Desember 2018 hingga 29 Mei 2025**. Fitur yang digunakan untuk pemodelan meliputi:

-   `Open`
-   `High`
-   `Low`
-   `Close`
-   `Volume`

Target prediksi (*label*) adalah harga penutupan (`Close`) saham pada hari berikutnya.

## Metodologi

Analisis dalam *notebook* ini mengikuti alur kerja standar untuk pemodelan *time-series* dengan optimasi hyperparameter:

1.  **Pengambilan dan Persiapan Data**: Data saham diunduh dan dibersihkan. Selanjutnya, data dibagi menjadi tiga set: **70% data latih**, **15% data validasi**, dan **15% data uji**.
2.  **Pra-pemrosesan**:
    -   **Normalisasi**: Fitur-fitur dinormalisasi menggunakan `MinMaxScaler` untuk mengubah skala nilai antara 0 dan 1, yang penting untuk stabilitas training model neural network.
    -   **Pembuatan Sekuens**: Data diubah menjadi format sekuens (3D array) dengan *time steps* sebanyak **60 hari**, sesuai dengan kebutuhan input model LSTM dan GRU.

3.  **Hyperparameter Tuning**:
    -   **Model Builder**: Fungsi `build_lstm_model()` dan `build_gru_model()` dibuat untuk mendefinisikan ruang pencarian (*search space*) hyperparameter yang akan diuji oleh Keras Tuner.
    -   **Tuner**: Metode `RandomSearch` dari **Keras Tuner** digunakan untuk secara otomatis mencoba 10 kombinasi hyperparameter yang berbeda pada kedua model.
    -   **Tujuan Optimasi**: Proses tuning bertujuan untuk meminimalkan *loss* pada data validasi (`val_loss`).

## Hasil dan Analisis

Proses *hyperparameter tuning* berhasil menemukan kombinasi parameter yang jauh lebih optimal dibandingkan dengan konfigurasi *default*.

### Konfigurasi Hyperparameter Terbaik

-   **LSTM Terbaik**:
    -   `units`: 256
    -   `activation`: tanh
    -   `dropout`: 0.1
    -   `optimizer`: rmsprop
    -   `learning_rate`: 0.001

-   **GRU Terbaik**:
    -   `units`: 32
    -   `activation`: tanh
    -   `dropout`: 0.0
    -   `optimizer`: adam
    -   `learning_rate`: 0.001

### Evaluasi Performa

Setelah dilatih kembali dengan hyperparameter terbaik, kedua model dievaluasi pada data uji:

| Model | RMSE | MAE |
| :--- | :--- | :--- |
| **LSTM (Tuned)** | 1408.00 | 1081.89 |
| **GRU (Tuned)** | 1241.18 | 927.58 |

## Kesimpulan

*Hyperparameter tuning* terbukti memberikan **peningkatan performa yang signifikan**. Dibandingkan dengan model dasar (tanpa tuning) dari tugas sebelumnya, nilai RMSE dan MAE dari kedua model berhasil diturunkan secara drastis, menunjukkan kesalahan prediksi yang lebih kecil.

-   **GRU** dengan hyperparameter yang telah dioptimalkan menunjukkan performa prediksi terbaik dengan nilai **RMSE 1241.18**.


Proyek ini menegaskan bahwa otomatisasi *hyperparameter tuning* adalah langkah krusial dalam membangun model *deep learning* yang andal dan akurat untuk data *time-series*.
