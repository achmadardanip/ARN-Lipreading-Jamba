import cv2
import os

def get_total_frames(video_path):
    """Mendapatkan jumlah total frame dari sebuah video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def display_frames_in_folder_recursive(root_folder_path):
    """Menampilkan total frame untuk setiap video di dalam folder dan semua subfoldernya."""
    print(f"Menganalisis video di folder dan subfolder dari: {root_folder_path}\n")
    found_videos_overall = False

    # os.walk() akan menjelajahi direktori secara rekursif
    # root: path direktori saat ini
    # dirs: daftar nama subdirektori di dalam root
    # files: daftar nama file di dalam root
    for root, dirs, files in os.walk(root_folder_path):
        for filename in files:
            # Tambahkan ekstensi video lain jika diperlukan (misalnya, '.avi', '.mov', '.wmv')
            if filename.lower().endswith(('.mp4', '.mkv', '.flv', '.mpeg', '.webm', '.avi', '.mov')):
                found_videos_overall = True
                video_path = os.path.join(root, filename) # Menggunakan 'root' untuk path yang benar
                total_frames = get_total_frames(video_path)
                if total_frames is not None:
                    print(f"Video: {video_path} - Total Frames: {total_frames}")
    
    if not found_videos_overall:
        print("Tidak ada file video yang ditemukan di folder ini atau subfoldernya.")

if __name__ == "__main__":
    # Ganti dengan path ke folder utama video Anda
    main_video_folder = input("Masukkan path ke folder utama video Anda: ") 
    
    if os.path.isdir(main_video_folder):
        display_frames_in_folder_recursive(main_video_folder)
    else:
        print(f"Error: Folder '{main_video_folder}' tidak ditemukan.")