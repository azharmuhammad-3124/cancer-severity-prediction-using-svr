# Cancer Severity Prediction with SVR

## Repository Outline
1. cancer_severity - Folder source code deployment
2. description.md - Penjelasan gambaran umum project
3. global_cancer_patients_2015_2024.csv - File dataset
4. P1M2_azhar_muhammad_conceptual - Pertanyaan dan jawaban
mengenai conceptual problems
5. P1M2_azhar_muhammad_inf.ipynb - Notebook berisi model inferencing
6. P1M2_azhar_muhammad.ipynb - Notebook utama berisi proses EDA, preprocessing, modelling, dan evaluasi
7. model.pkl - File model hasil pelatihan untuk inference
8. url.txt - URL untuk model deployment dan source dataset

## Problem Background
Biaya dan dampak sosial akibat penyakit kanker sangat besar, terutama jika tidak ditangani secara tepat waktu. Untuk membantu proses pengambilan keputusan medis, diperlukan sistem prediktif yang dapat mengestimasi tingkat keparahan pasien berdasarkan data karakteristik pasien. Project ini bertujuan membangun model machine learning untuk memprediksi tingkat keparahan kanker berdasarkan data pasien dari berbagai negara.

## Project Output
Output dari project ini adalah sebuah model machine learning berbasis Support Vector Regression (SVR) yang mampu memprediksi tingkat keparahan kanker (target_severity_score) berdasarkan berbagai fitur pasien. Model telah disimpan dalam format .pkl dan siap digunakan untuk inference. Selain itu, model ini juga dapat di-deploy atau diunggah ke platform Hugging Face untuk kebutuhan demonstrasi.

## Data
Dataset terdiri dari ribuan entri pasien kanker dari berbagai negara. Data mencakup fitur-fitur seperti: jenis kanker, stadium, gender, usia, biaya pengobatan, serta faktor risiko seperti merokok, alkohol, dan polusi udara.

- Jumlah fitur: 15 kolom, 50000 baris
- Tidak ada missing values dan outlier
- Distribusi data sangat bersih dan seimbang, kemungkinan telah melalui preprocessing sebelumnya

## Method
Proyek ini menggunakan pendekatan supervised learning untuk prediksi nilai kontinu dari skor tingkat keparahan kanker (target_severity_score). Model utama yang digunakan adalah Support Vector Regression (SVR). Proses modeling mencakup:

- Exploratory Data Analysis (EDA)
- Feature engineering menggunakan korelasi phik
- Preprocessing dengan StandardScaler dan PCA
- Evaluasi awal menggunakan cross-validation dan metrik MAE, RMSE, dan R²
- Hyperparameter tuning menggunakan RandomizedSearchCV
- Perbandingan performa model KNN, SVR, Decision Tree, Random Forest, dan AdaBoost

## Stacks
Bahasa Pemrograman: Python

Libraries & Tools:

- Data processing: pandas, numpy
- Visualisasi: matplotlib, seaborn
- Feature correlation: phik
- Machine Learning: scikit-learn (SVR, KNN, DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor)
- Preprocessing: StandardScaler, PCA, ColumnTransformer, Pipeline
- Model evaluation & tuning: train_test_split, KFold, cross_val_score, RandomizedSearchCV, mean_absolute_error, mean_squared_error, r2_score
- Model saving: pickle

## Reference
[`URL Deployment`](https://huggingface.co/spaces/azhar-muhammad/cancer_severity_app)

## Note:
Kadang aplikasi tidak bisa dibuka di hugging face, muncul error: 'Error: not connected to a server!'. Source Code-nya sudah aman, di run local tidak ada problem sama sekali.
---
