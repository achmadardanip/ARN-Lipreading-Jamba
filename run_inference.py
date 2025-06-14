# run_inference.py
import modal
import os
import sys
import subprocess
import typer
from pathlib import Path # Tetap import Path
from modal import FilePatternMatcher # Import yang diperlukan
import shutil

# --- Konfigurasi Stub Modal ---
APP_NAME = "arn-lipreading-inference-app"
stub = modal.App(APP_NAME)

# --- Konfigurasi Network File System (NFS) ---
# NFS_NAME = "lipreading-dataset-nfs"
# volume = modal.NetworkFileSystem.from_name(NFS_NAME, create_if_missing=False)

# --- Path di dalam NFS Modal ---
# --- Path ---
# Path di dalam container tempat NFS di-mount
# MODAL_NFS_MOUNT_PATH = "/nfs"
# Path remote di root NFS (untuk .create_dir dan .write_file)
# NFS_REMOTE_OUTPUT_DIR = "/outputs"
# NFS_REMOTE_INFERENCE_DIR = "/inference_data"
# Path lengkap di dalam container (untuk diakses oleh skrip)
# NFS_CONTAINER_OUTPUT_PATH = os.path.join(MODAL_NFS_MOUNT_PATH, NFS_REMOTE_OUTPUT_DIR.lstrip('/'))
# NFS_CONTAINER_INFERENCE_PATH = os.path.join(MODAL_NFS_MOUNT_PATH, NFS_REMOTE_INFERENCE_DIR.lstrip('/'))


## BARU MENGGUNAKAN MODAL VOLUME

# --- PERBAIKAN: Gunakan modal.Volume ---
VOLUME_NAME = "arn-lipreading-volume"
# from_name akan mencari Volume yang ada, atau membuatnya jika tidak ada
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- Path ---
# Path di dalam container tempat Volume di-mount
MODAL_VOLUME_MOUNT_PATH = "/data_vol"
VOLUME_OUTPUT_PATH = os.path.join(MODAL_VOLUME_MOUNT_PATH, "outputs")
VOLUME_INFERENCE_DATA_PATH = os.path.join(MODAL_VOLUME_MOUNT_PATH, "inference_data")

# Gunakan image yang sama dengan versi 'dijamin berhasil' dari training
# app_image = (
#     modal.Image.debian_slim(python_version="3.10")
#     .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libjpeg-turbo8", "libjpeg-turbo8-dev")
#     .pip_install(
#         "protobuf==3.20.3", "torch==2.7.0", "torchvision==0.22.0", "torchaudio==0.27.0",
#         "numpy==2.0.2", "opencv-python==4.11.0.86", "PyTurboJPEG==1.8.0",
#         "transformers==4.52.4", "accelerate", "bitsandbytes", "packaging",
#         "scikit-learn", "pandas", "aiohttp", "requests", "tqdm",
#         "fightingcv-attention", "tensorboardX", "focal-loss-torch",
#         "lion-pytorch", "torch-optimizer", "calflops", "fvcore",
#         "pytorch-model-summary", "typer"
#     )
#     .add_local_dir(local_path=".", remote_path="/root")
# )

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


# --- PERBAIKAN: Fungsi menerima konten video (bytes), bukan nama file ---
@stub.function(
    image=app_image,
    gpu="H100",
    volumes={MODAL_VOLUME_MOUNT_PATH: volume},
    timeout=300
)
def run_inference_on_modal(weights_name: str, video_content: bytes):
    if "/root" not in sys.path: sys.path.append("/root")

    # Simpan konten video yang diterima ke file sementara di dalam container
    local_video_path = "/tmp/inference_video.mp4"
    with open(local_video_path, "wb") as f:
        f.write(video_content)

    weights_path = os.path.join(VOLUME_OUTPUT_PATH, weights_name)
    
    print(f"--- Menjalankan Inferensi ---")
    print(f"Model: {weights_path}")
    print(f"Video (disimpan sementara di): {local_video_path}")
    
    # Jalankan inference.py menggunakan path lokal sementara
    command = ["python", "inference.py", "--weights", weights_path, "--video", local_video_path]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Inferensi Gagal (exit code: {e.returncode}) !!!")
        raise

# --- PERBAIKAN: Entrypoint membaca file lokal dan mengirim kontennya ---
@stub.local_entrypoint()
def main(
    video_path: str = typer.Argument(..., help="Path ke file video LOKAL."),
    weights: str = typer.Option(..., "--weights", help="Nama file model (.pt) di folder 'outputs' Volume."),
):
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"❌ Error: File video tidak ditemukan di '{video_path_obj}'")
        return

    # Baca konten file video sebagai data biner (bytes)
    with open(video_path_obj, "rb") as f:
        video_bytes = f.read()
    
    print(f"🚀 Mengirim konten video '{video_path_obj.name}' dan menjalankan inferensi di Modal...")
    
    # Panggil fungsi remote dengan mengirim konten video
    run_inference_on_modal.remote(weights, video_bytes)