# ARN-Lipreading-Jamba: Lip Reading Bahasa Indonesia dengan Jamba

Repositori ini berisi implementasi model *deep learning* untuk pengenalan gerak bibir (lip-reading) tingkat kata dalam Bahasa Indonesia. Arsitektur yang digunakan adalah **ARN-Lipreading-Jamba**, sebuah model hybrid yang menggabungkan CNN (ResNet dengan atensi), Bi-GRU, dan arsitektur modern Jamba (Transformer-Mamba).

Proyek ini juga mencakup **LIRA-Gen**, sebuah *tool* untuk membangun dataset *lip-reading* dari video, serta dataset **IDLRW** yang dihasilkan darinya.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## Fitur Utama

- **Arsitektur Hybrid Inovatif**: Menggabungkan kekuatan ResNet untuk ekstraksi fitur visual, Bi-GRU untuk pemodelan sekuensial, dan Jamba untuk menangkap dependensi temporal yang kompleks.
- **Dataset Bahasa Indonesia**: Menyediakan dataset `IDLRW` yang merupakan salah satu dataset *lip-reading* pertama untuk Bahasa Indonesia.
- **Eksekusi di Cloud**: Dilengkapi dengan skrip untuk menjalankan *training* dan inferensi secara efisien di [Modal.com](https://modal.com) menggunakan GPU H100.
- **Modular**: Kode dipisahkan dengan jelas antara definisi model, utilitas dataset, dan skrip eksekusi.

## Struktur Repositori

```
/
├── model/                  # Definisi arsitektur model (VideoModel, Jamba, dll.)
│   ├── model.py
│   └── video_cnn.py
├── utils/                  # Skrip untuk dataset dan transformasi (IDLRWDataset, dll.)
│   ├── dataset_idlrw.py
│   └── cvtransforms.py
├── scripts/                # Skrip untuk pra-pemrosesan data (LIRA-Gen)
│   └── prepare_idev1.py
├── main_visual.py          # Skrip utama untuk training dan testing lokal
├── inference.py            # Skrip untuk menjalankan prediksi pada satu video
├── run_modal.py            # Skrip untuk menjalankan training/testing di Modal
├── run_inference.py        # Skrip untuk menjalankan inferensi di Modal
├── requirements.txt        # Daftar dependensi Python
└── .modalignore            # File untuk mengabaikan direktori saat deploy ke Modal
```

## Instalasi & Penyiapan Lokal

1.  **Clone repositori ini:**
    ```bash
    git clone [URL-Git-Anda]
    cd [nama-repositori-Anda]
    ```

2.  **Buat dan aktifkan Virtual Environment:**
    ```bash
    # Buat venv
    python -m venv venv

    # Aktifkan di Windows (PowerShell)
    .\venv\Scripts\Activate.ps1

    # Aktifkan di macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependensi:**
    File `requirements.txt` di bawah ini adalah versi yang sudah bersih dan terbukti berhasil.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Siapkan Dataset IDLRW yang sudah berbentuk .pkl**:
    Unduh dataset IDLRW (atau buat dengan LIRA-Gen) dan letakkan di dalam direktori dengan struktur berikut:
    ```
    /data/idev1_roi_80_116_175_211_npy_gray_pkl_jpeg/
    ├── ada/
    │   ├── train/
    │   │   ├── ada_00001.pkl
    │   │   └── ...
    │   └── test/
    │       ├── ada_00002.pkl
    │       └── ...
    ├── dan/
    │   ├── train/
    │   └── test/
    └── ...
    ```

## Cara Penggunaan (Training & Inferensi di Modal.com)

Metode ini direkomendasikan untuk memanfaatkan GPU H100.

1.  **Penyiapan Modal**:
    * Install Modal: `pip install modal`
    * Lakukan setup awal: `modal setup`

2.  **Unggah Dataset ke Modal NFS**:
    Pastikan direktori data Anda (langkah instalasi #4) sudah diunggah ke Modal NFS dengan nama `lipreading-dataset-nfs`.
    ```bash
    # Perintah ini hanya perlu dijalankan sekali
    modal nfs put lipreading-dataset-nfs /path/local/ke/idev1_roi_80_116_175_211_npy_gray_pkl_jpeg /data/
    ```

3.  **Menjalankan Training**:
    Untuk memulai proses *training* model dari awal di Modal, jalankan perintah:
    ```bash
    modal run run_modal.py --action train
    ```
    Model terbaik akan secara otomatis disimpan di folder `outputs` pada NFS Anda.

4.  **Menjalankan Evaluasi (Testing)**:
    Setelah training selesai, Anda bisa menguji model terbaik Anda. Ganti nama file `.pt` sesuai dengan hasil training Anda.
    ```bash
    modal run run_modal.py --action test --weights jamba_lipreading_model_best_acc_0.2973.pt
    ```

5.  **Menjalankan Inferensi pada Video Baru**:
    Untuk mencoba prediksi pada satu file video, gunakan skrip `run_inference.py`.
    ```bash
    # Ganti 'video_tes.mp4' dengan path video lokal Anda
    modal run run_inference.py --video-path video_tes.mp4 --weights jamba_lipreading_model_best_acc_0.2973.pt
    ```

## Hasil Awal (Baseline)

Evaluasi pada *test set* menggunakan model yang telah dilatih pada 76 sampel data menghasilkan performa awal sebagai berikut. Hasil ini menunjukkan bahwa model sudah mulai belajar namun membutuhkan lebih banyak data untuk meningkatkan performa.

* **Accuracy**: ~29.7%
* **F1 Score**: ~16.3%
* **Total Parameters**: 72.12 Juta
* **FLOPs**: ~34.38 GFLOPS

## Rencana Peningkatan

* **Penambahan Data**: Langkah paling krusial adalah menambah jumlah data training secara signifikan untuk setiap kelas kata.
* **Hyperparameter Tuning**: Melakukan eksperimen dengan *learning rate*, *batch size*, dan jumlah epoch yang berbeda.
* **Eksplorasi Arsitektur**: Mencoba variasi lain dari rasio lapisan Transformer dan Mamba di dalam Jamba.

## Sitasi

Jika Anda menggunakan proyek ini, mohon pertimbangkan untuk mengutip paper-paper relevan yang menjadi dasar pekerjaan ini.
```
@article{Jamba,
  title={Jamba: A Hybrid Transformer-Mamba Language Model},
  author={Opher Lieber, et al.},
  journal={arXiv preprint arXiv:2403.19887},
  year={2024}
}

@article{Rahmatullah2025,
  title={Recognizing Indonesian words based on visual cues of lip movement using deep learning},
  author={Rahmatullah, Griffani Megiyanto and Ruan, Shanq-Jang and Li, Lieber Po-Hung},
  journal={Measurement},
  volume={250},
  pages={116968},
  year={2025},
  publisher={Elsevier}
}
```
