# encoding: utf-8
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import torch

import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch

jpeg = TurboJPEG()
def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            frame = frame[115:211, 79:175]
            frame = jpeg.encode(frame)
            video.append(frame)
        else:
            break
    cap.release()
    return video        


target_dir = 'idev1_roi_80_116_175_211_npy_gray_pkl_jpeg'

if(not os.path.exists(target_dir)):
    os.makedirs(target_dir)    

class idev1Dataset(Dataset):
    def __init__(self):

        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('idev1', label, '*', '*.mp4'))
            for file in files:
                savefile = file.replace('idev1', target_dir).replace('.mp4', '.pkl')
                savepath = os.path.split(savefile)[0]
                if(not os.path.exists(savepath)):
                    os.makedirs(savepath)
                
            files = sorted(files)
            

            self.list += [(file, i) for file in files]                                                                                
            
        
    def __getitem__(self, idx):
            
        inputs = extract_opencv(self.list[idx][0])
        result = {}        
         
        name = self.list[idx][0]
        duration = self.list[idx][0]            
        labels = self.list[idx][1]

                    
        result['video'] = inputs
        result['label'] = int(labels)
        result['duration'] = self.load_duration(duration.replace('.mp4', '.txt')).astype(np.bool)
        savename = self.list[idx][0].replace('idev1', target_dir).replace('.mp4', '.pkl')
        torch.save(result, savename)
        
        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = np.zeros(25)
        mid = 25 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor            

if(__name__ == '__main__'):
        # Buat instance dataset terlebih dahulu
    dataset = idev1Dataset()
    
    # Periksa dan cetak ukuran dataset
    print(f"✅ Total item yang ditemukan di dataset: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ Dataset kosong! Loop tidak akan berjalan. Periksa path dan struktur folder Anda.")
        exit() # Keluar dari script jika dataset kosong
        
    loader = DataLoader(idev1Dataset(),
            batch_size = 1, 
            num_workers = 16,   
            shuffle = False,         
            drop_last = False)
    
    import time
    tic = time.time()
    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')        