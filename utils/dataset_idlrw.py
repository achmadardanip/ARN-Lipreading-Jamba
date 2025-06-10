# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from .cvtransforms import *
import torch
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE


jpeg = TurboJPEG()
class IDLRWDataset(Dataset):
    def __init__(self, phase, args, data_path: str):

        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []
        self.unlabel_list = []
        self.phase = phase        
        self.args = args
        
        if(not hasattr(self.args, 'is_aug')):
            setattr(self.args, 'is_aug', True)

        # for (i, label) in enumerate(self.labels):
         
        #     # files = glob.glob(os.path.join('idev1_roi_64_160_64_160_npy_gray_pkl_jpeg_with_border', label, phase, '*.pkl'))
        #     files = glob.glob(os.path.join('data_path', label, phase, '*.pkl'))                     
        #     files = sorted(files)
            

        #     self.list += [file for file in files]

        for (i, label) in enumerate(self.labels):
            # Pola path yang benar: data_path/{label}/{phase}/*.pkl
            search_path = os.path.join(data_path, label, phase, '*.pkl')
            files = glob.glob(search_path)
            
            if i < 2: # Hanya print untuk 2 label pertama agar log tidak terlalu panjang
                 print(f"    Mencari di: {search_path} -> Ditemukan {len(files)} file.")

            if files:
                self.list.extend(files)
        
        print(f"--> [IDLRWDataset] Total file ditemukan: {len(self.list)}")
            
        
    def __getitem__(self, idx):
            
        # tensor = torch.load(self.list[idx])    

        tensor = torch.load(self.list[idx], weights_only=False)                
        
        inputs = tensor.get('video')
        inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs] # jadi grayscale
        inputs = np.stack(inputs, 0) / 255.0 # jadi tensor 0-1
        inputs = inputs[:,:,:,0]

        # --- PERBAIKAN: Standarisasi jumlah frame video ---
        target_frames = 25
        num_frames = inputs.shape[0]

        if num_frames > target_frames:
            # Potong (trim) secara acak
            start = random.randint(0, num_frames - target_frames)
            inputs = inputs[start : start + target_frames, :, :]
        elif num_frames < target_frames:
            # Tambah (pad) dengan duplikasi frame terakhir
            padding = np.tile(inputs[-1, :, :], (target_frames - num_frames, 1, 1))
            inputs = np.concatenate([inputs, padding], axis=0)
        # --- PERBAIKAN SELESAI ---
                
        if(self.phase == 'train'):
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = HorizontalFlip(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = CenterCrop(inputs, (88, 88))
        
        result = {}            
        result['video'] = torch.FloatTensor(batch_img[:,np.newaxis,...])
        #print(result['video'].size())
        result['label'] = tensor.get('label')
        #sementara karena tidak pakai duration
        result['duration'] = 1.0 * tensor.get('duration')

        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = torch.zeros(25)
        mid = 25 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor            
        