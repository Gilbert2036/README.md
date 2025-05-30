# Laporan Proyek Machine Learning - Gilbert O.K.D. Malau

## Domain Proyek
Seiring dengan pertumbuhan industri otomotif dan meningkatnya minat masyarakat terhadap kendaraan pribadi, kebutuhan akan informasi harga mobil yang akurat dan transparan menjadi semakin penting. Dalam proses jual beli mobil, baik baru maupun bekas, salah satu tantangan utama adalah menentukan harga pasar yang wajar berdasarkan berbagai faktor seperti tahun pembuatan, jarak tempuh (mileage), kapasitas mesin (engine size), jenis bahan bakar, serta kondisi kendaraan. Kesalahan dalam memperkirakan harga dapat merugikan penjual maupun pembeli, serta menciptakan ketidakseimbangan pasar.

## Business Understanding
Dalam industri otomotif, khususnya dalam pasar mobil bekas, menentukan harga jual yang tepat merupakan tantangan yang signifikan. Harga mobil dipengaruhi oleh berbagai faktor seperti tahun pembuatan, kilometer yang telah ditempuh, jenis bahan bakar, jenis transmisi, serta kondisi umum kendaraan. Ketidakakuratan dalam menentukan harga dapat menyebabkan kerugian finansial bagi penjual ataupun pembeli, serta memperlambat proses jual beli.

### Problem Statements
Pernyataan Masalah 1:
Bagaimana cara mengidentifikasi fitur-fitur kendaraan yang paling berpengaruh terhadap harga jual mobil bekas, seperti tahun pembuatan, kilometer tempuh, jenis bahan bakar, dan tipe transmisi?

Pernyataan Masalah 2:
Bagaimana cara melakukan praproses data secara efektif untuk mengatasi data yang tidak lengkap, data kategorikal, dan outlier yang dapat mempengaruhi performa model prediksi?

Pernyataan Masalah 3:
Model machine learning regresi apa yang paling optimal untuk memprediksi harga mobil bekas berdasarkan fitur-fitur yang tersedia dalam dataset CarDekho?

### Goals
Jawaban Pernyataan Masalah 1:
Mengidentifikasi dan menganalisis fitur-fitur penting yang paling memengaruhi harga jual mobil bekas melalui eksplorasi data (EDA) dan analisis korelasi. Dengan ini, kita dapat memahami variabel mana yang berkontribusi signifikan terhadap nilai akhir mobil.

Jawaban Pernyataan Masalah 2:
Melakukan proses pembersihan dan praproses data secara menyeluruh, termasuk penanganan missing values, encoding variabel kategorikal, normalisasi data numerik, serta deteksi dan penanganan outlier agar data siap digunakan dalam pelatihan model machine learning.

Jawaban Pernyataan Masalah 3:
Membangun dan melatih berbagai model regresi (Linear Regression, Random Forest, XGBoost, dll.) untuk menemukan model dengan kinerja terbaik dalam memprediksi harga mobil berdasarkan data historis yang tersedia.

 ### Solution statements
Solusi 1: 
Membangun dan Membandingkan Beberapa Model Regresi
Membangun beberapa model machine learning regresi seperti:
-Linear Regression sebagai baseline model
-Random Forest Regressor untuk menangani non-linearitas dan fitur penting
-XGBoost Regressor sebagai model lanjutan dengan kemampuan generalisasi yang lebih baik

Solusi 2: 
Feature Engineering dan Seleksi Fitur
Melakukan transformasi fitur dan seleksi fitur untuk meningkatkan relevansi dan kualitas data yang digunakan oleh model, seperti:
Encoding fitur kategorikal (Fuel Type, Transmission, dll.)
Feature scaling pada data numerik (Mileage, Engine, dll.)
Menghapus fitur dengan korelasi rendah terhadap harga jual

Solusi 3: 
Evaluasi Kinerja Model Menggunakan Mean Squared Error (MSE)
Melakukan evaluasi kuantitatif terhadap setiap model machine learning menggunakan Mean Squared Error (MSE) yang telah disesuaikan satuannya (dibagi 1000 atau 1e3 untuk skala interpretasi yang lebih baik). MSE dihitung pada data training dan testing untuk mengukur performa model dan mendeteksi potensi overfitting atau underfitting.

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan dataset mobil bekas yang diperoleh dari platform CarDekho, salah satu situs marketplace otomotif terbesar di India. Dataset ini berisi informasi mengenai spesifikasi kendaraan serta harga jual mobil bekas yang tercatat. Dataset ini tersedia secara publik dan dapat diunduh melalui tautan berikut:
[🔗 Kaggle - CarDekho Vehicle Dataset]
Deskripsi Dataset
Dataset ini terdiri dari beberapa fitur penting yang dapat digunakan untuk memprediksi harga jual mobil bekas. Berikut adalah uraian masing-masing fitur:

## Deskripsi Dataset

| Nama Fitur       | Tipe Data   | Deskripsi                                                                 |
|------------------|-------------|---------------------------------------------------------------------------|
| `Car_Name`       | Kategorikal | Nama mobil (brand + model). Umumnya tidak digunakan langsung dalam model.|
| `Year`           | Numerik     | Tahun mobil dibuat (digunakan untuk menghitung usia kendaraan).          |
| `Selling_Price`  | Numerik     | Harga jual mobil (dalam lakhs/100,000 INR) – *target variable*.          |
| `Present_Price`  | Numerik     | Harga mobil saat ini jika baru (dalam lakhs).                            |
| `Kms_Driven`     | Numerik     | Jumlah kilometer yang telah ditempuh oleh mobil.                         |
| `Fuel_Type`      | Kategorikal | Jenis bahan bakar: Petrol, Diesel, CNG.                                  |
| `Seller_Type`    | Kategorikal | Tipe penjual: Individual, Dealer, atau Trustmark Dealer.                |
| `Transmission`   | Kategorikal | Jenis transmisi: Manual atau Automatic.                                  |
| `Owner`          | Numerik     | Jumlah kepemilikan sebelumnya: 0, 1, 2, atau 3.                          |

## Data Preparation
  1. One-Hot Encoding untuk Variabel Kategorikal
Langkah:
Mengubah variabel kategorikal menjadi variabel numerik menggunakan teknik One-Hot Encoding:
fuel
seller_type
transmission
owner

1. Principal Component Analysis (PCA)
Langkah:
PCA dilakukan pada fitur numerik 'year','km_driven','mileage(km/ltr/kg)' untuk mengurangi dimensi menjadi 3 komponen utama.

3. Memisahkan Fitur dan Target
Langkah:
Memisahkan fitur prediktor (X) dan target variabel (y) yaitu selling_price

5. Train-Test Split
Langkah:
Membagi dataset menjadi data pelatihan (90%) dan data pengujian (10%) dengan seed random untuk replikasi.

7. Feature Scaling pada Fitur Numerik
Langkah:
Standarisasi fitur numerik: year, km_driven, mileage(km/ltr/kg), dan dimension menggunakan StandardScaler. Beberapa model regresi sangat sensitif terhadap skala fitur. Proses standardisasi akan membuat mean = 0 dan standar deviasi = 1 untuk menghindari dominasi fitur berskala besar.

## Modeling
Pada tahap ini, beberapa algoritma regresi digunakan untuk membangun model prediksi harga mobil berdasarkan fitur-fitur yang telah diproses sebelumnya. Tujuan utamanya adalah memilih model terbaik berdasarkan performa pada data uji, khususnya dengan menggunakan metrik Mean Squared Error (MSE) yang dibagi 1000 untuk efisiensi interpretasi.
### Hasil Evaluasi Model (MSE)

| Model      | Train MSE       | Test MSE        |
|------------|------------------|------------------|
| KNN        | 8978007.43     | 11471841.50    |
| RandomForest (RF) | 1397501.25     | **6077409.28**    |
| AdaBoost (Boosting) | 19216221.72    | 18590823.96    |

Nilai MSE yang lebih kecil menandakan performa prediksi yang lebih baik.
Random Forest menunjukkan performa terbaik pada data uji (test set), dengan generalisasi yang sangat baik tanpa overfitting.


 




