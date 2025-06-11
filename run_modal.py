import modal
import os
import sys
import subprocess
import traceback
from modal import FilePatternMatcher # Import yang diperlukan
import typer

# --- Konfigurasi Stub Modal ---
APP_NAME = "arn-lipreading-jamba-app"
stub = modal.App(APP_NAME)

# --- Konfigurasi Network File System (NFS) ---
# NFS_NAME = "lipreading-dataset-nfs"
# volume = modal.NetworkFileSystem.from_name(NFS_NAME, create_if_missing=True)

# --- Path di dalam NFS Modal ---
# Ini akan menjadi mount point untuk NFS di dalam container
# MODAL_NFS_MOUNT_PATH = "/nfs"

# NFS_DATASET_PATH = os.path.join(MODAL_NFS_MOUNT_PATH, "data/idev1_roi_80_116_175_211_npy_gray_pkl_jpeg")
# NFS_OUTPUT_PATH = os.path.join(MODAL_NFS_MOUNT_PATH, "outputs")

# baru menggunakan modal volume

VOLUME_NAME = "arn-lipreading-volume"
# Volume.from_name akan mencari volume yang ada, atau membuatnya jika tidak ada
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- Path ---
MODAL_VOLUME_MOUNT_PATH = "/data_vol" # Ganti nama mount point agar lebih jelas
VOLUME_DATASET_PATH = os.path.join(MODAL_VOLUME_MOUNT_PATH, "data/idev1_roi_80_116_175_211_npy_gray_pkl_jpeg")
VOLUME_OUTPUT_PATH = os.path.join(MODAL_VOLUME_MOUNT_PATH, "outputs")

# --- Definisi Image Docker untuk Environment ---
# Menggunakan API modern .add_local_dir() yang secara otomatis menghormati .modalignore


## versi lama tanpa mamba (ORIGINAL JANGAN DIUBAH)
# app_image = (
#     modal.Image.debian_slim(python_version="3.10")
#     .apt_install(
#         "libgl1-mesa-glx",
#         "libglib2.0-0",
#         "libturbojpeg0",
#         "libturbojpeg0-dev"
#     )
#     .pip_install_from_requirements("requirements.txt")
#     .add_local_dir(".", remote_path="/root", ignore=FilePatternMatcher.from_file(".modalignore"))
# )

# --- Definisi Image dengan Instalasi Bertahap (TIDAK MENGGUNAKAN requirements.txt) ---
app_image = (
    # 1. Mulai dari base image Nvidia dan tambahkan Python 3.10
        modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10"
    )
    .env({"CC": "gcc", "CXX": "g++", "PIP_NO_BUILD_ISOLATION": "1"})
    # 2. Instal perangkat sistem yang dibutuhkan
    .apt_install(
        "git",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libjpeg-turbo8",
        "libjpeg-turbo8-dev",
        "libturbojpeg0-dev"
    )
    # 3. Instal build-tools untuk Python
    .pip_install("ninja", "wheel", "packaging", "numpy==2.0.2")
    # 4. Instal PyTorch secara terpisah
    .pip_install("torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0")
    # 5. Instal kernel kustom dengan flag --no-build-isolation
    .run_commands([
        "pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0.post8",
        "pip install --no-build-isolation git+https://github.com/state-spaces/mamba.git@v2.2.4"
    ])
    # 6. Instal sisa dependensi lainnya
    .pip_install(
        "protobuf==3.20.3", "opencv-python==4.11.0.86", "PyTurboJPEG==1.8.0",
        "transformers==4.52.4", "accelerate", "bitsandbytes", "scikit-learn", "pandas",
        "aiohttp", "requests", "tqdm", "fightingcv-attention", "tensorboardX",
        "focal-loss-torch", "lion-pytorch", "torch-optimizer", "calflops", "fvcore",
        "pytorch-model-summary", "fightingcv-attention", "typer"
    )
    .add_local_dir(
        local_path=".",
        remote_path="/root",
        ignore=FilePatternMatcher.from_file(".modalignore")
    )
)


# --- Fungsi Training yang akan dijalankan di Modal ---
@stub.function(
    image=app_image,
    gpu="H100",
    # network_file_systems={MODAL_NFS_MOUNT_PATH: volume}, # Mount NFS ke /nfs
    # PERBAIKAN: Gunakan 'volumes' bukan 'network_file_systems'
    volumes={MODAL_VOLUME_MOUNT_PATH: volume},
    timeout=3600 * 6,  # Timeout 6 jam
    secrets=[modal.Secret.from_dict({"PYTHONUNBUFFERED": "1"})], # Untuk log real-time
)
def train_model():
    # Pastikan direktori kode ada di path Python
    if "/root" not in sys.path:
        sys.path.append("/root")

    import torch

    print("--- Memulai Training ARN-Lipreading di Modal ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA tersedia: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Versi CUDA: {torch.version.cuda}")
        print(f"Nama GPU: {torch.cuda.get_device_name(0)}")

    # Membuat direktori output di NFS
    # Nama model akan digabungkan dengan path ini oleh skrip main_visual.py
    # os.makedirs(NFS_OUTPUT_PATH, exist_ok=True)
    #BARU PAKE MODAL VOLUME
    os.makedirs(VOLUME_OUTPUT_PATH, exist_ok=True)
    print(f"Direktori output dipastikan ada di Volume: {VOLUME_OUTPUT_PATH}")

    # Definisikan perintah untuk menjalankan main_visual.py
    command = [
        "python", "main_visual.py",
        "--gpus", "0",
        "--lr", "1e-4",
        "--batch_size", "32",
        "--n_class", "100",
        "--num_workers", "4",
        "--max_epoch", "1000",
        "--test", "false",
        "--save_prefix", os.path.join(VOLUME_OUTPUT_PATH, "jamba_lipreading_model"),
        "--dataset", "idlrw",
        "--data_path", VOLUME_DATASET_PATH,
        "--border", "true",
        "--mixup", "true",
        "--label_smooth", "true",
        "--se", "true",
        "--focal_loss", "true",
        "--focal_loss_weight", "false"
    ]

    print("\nPerintah yang akan dieksekusi:")
    print(" ".join(command))
    print("-" * 20)

    try:
        # Jalankan proses training dengan penanganan error yang lebih baik
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Stream output log secara real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()

        if process.returncode == 0:
            print("\n--- Training ARN-Lipreading di Modal Selesai dengan Sukses ---")
        else:
            print(f"\n!!! Training ARN-Lipreading di Modal Gagal (exit code: {process.returncode}) !!!")

    except Exception as e:
        print(f"\n!!! Terjadi Error Kritis saat Menjalankan Training: {e} !!!")
        traceback.print_exc()
        raise # Tampilkan error ke Modal agar job ditandai gagal


# --- FUNGSI BARU: Untuk Testing ---
@stub.function(
    image=app_image,
    gpu="H100", # Bisa diganti ke GPU lebih kecil seperti T4 untuk menghemat biaya
    # network_file_systems={MODAL_NFS_MOUNT_PATH: volume},
        # PERBAIKAN: Gunakan 'volumes'
    volumes={MODAL_VOLUME_MOUNT_PATH: volume},
    timeout=600, # Testing biasanya cepat
)
def test_model(weights_name: str):
    """
    Menjalankan evaluasi model pada test set menggunakan checkpoint tertentu.
    """
    if "/root" not in sys.path: sys.path.append("/root")
    
    # weights_path = os.path.join(NFS_OUTPUT_PATH, weights_name)
    # BARU PAKE MODAL VOLUME
    weights_path = os.path.join(VOLUME_OUTPUT_PATH, weights_name)
    print(f"--- Memulai Evaluasi Model di Modal ---")
    print(f"Menggunakan bobot model dari: {weights_path}")

    command = [
        "python", "main_visual.py",
        "--gpus", "0",
        "--dataset", "idlrw",
        "--test", "true",
        "--weights", weights_path, # Path ke model di NFS
        "--data_path", VOLUME_DATASET_PATH,
        "--batch_size", "32",
        "--n_class", "100",
        "--num_workers", "4",
        "--border", "true",
        "--se", "true",
        "--mixup", "false",
        "--label_smooth", "false",
        "--focal_loss", "false",
        # Argumen dummy yang dibutuhkan oleh script
        "--lr", "0.0001",
        "--max_epoch", "1",
        "--save_prefix", "./test_run",
        "--focal_loss_weight", "false",
    ]

    print("\nPerintah yang akan dieksekusi:", " ".join(command))
    try:
        # Menggunakan subprocess.run karena testing adalah proses sinkron yang lebih pendek
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("\n--- Hasil Evaluasi ---")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Evaluasi Gagal (exit code: {e.returncode}) !!!")
        print("\n--- Output & Error Log ---")
        print(e.stdout)
        print(e.stderr)
        raise

# --- Entrypoint Lokal untuk Menjalankan Fungsi dari CLI ---
# @stub.local_entrypoint()
# def main():
#     print(f"Nama NFS yang akan digunakan: {NFS_NAME}")
#     print(f"Dataset diharapkan ada di path NFS: '/data/idev1_roi_80_116_175_211_npy_gray_pkl_jpeg'")
#     print(f"Output akan disimpan di path NFS: '/outputs'")
#     print("\nMemanggil fungsi training di Modal...")
    
#     train_model.remote()
    
#     print("\nFungsi training telah dipanggil. Periksa dashboard Modal untuk melihat log dan status.")


# --- ENTRYPOINT BARU: Bisa memilih aksi 'train' atau 'test' ---
@stub.local_entrypoint()
def main(
    action: str = typer.Argument(..., help="Aksi yang akan dijalankan: 'train' atau 'test'."),
    weights: str = typer.Option(None, "--weights", help="Nama file model (.pt) untuk testing."),
):
    """
    Entrypoint utama untuk menjalankan training atau testing.
    """
    if action == "train":
        print("Memanggil fungsi training di Modal...")
        train_model.remote()
    elif action == "test":
        if not weights:
            print("❌ Error: Untuk aksi 'test', Anda harus menyediakan nama file model dengan --weights <nama_file>")
            print("Contoh: modal run run_modal.py test --weights jamba_lipreading_model_best_acc_0.2973.pt")
            return
        print(f"Memanggil fungsi testing di Modal dengan model: {weights}")
        test_model.remote(weights)
    else:
        print(f"Aksi tidak dikenal: '{action}'. Pilihan yang tersedia: 'train', 'test'.")

