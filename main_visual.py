import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
import numpy as np
import time
from model import *
import torch.optim as optim 
import random
import pdb
import shutil
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter 
from focal_loss.focal_loss import FocalLoss
# from focallossv1 import *
# from focalloss_scratch import *
from lion_pytorch import Lion
import torch_optimizer as optim2
from calflops import calculate_flops        
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from pytorch_model_summary import summary
import pandas as pd
from fvcore.nn import FlopCountAnalysis

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()

# tensorboard
writer = SummaryWriter()

# set all random seed to 42
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
g = torch.Generator()
g.manual_seed(42)

def get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls, power=1):
    weights_for_samples = 1.0 / np.array(np.power(samples_per_cls, power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples

def get_weights_effective_num_of_samples(no_of_classes, beta, samples_per_cls):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights_for_samples = (1.0 - beta) / np.array(effective_num)
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples

def get_weights_transformed_for_sample(sample_weighting_method, no_of_classes, samples_per_cls, b_labels, beta=None):
    if sample_weighting_method == 'ens':
        weights_for_samples = get_weights_effective_num_of_samples(no_of_classes, beta, samples_per_cls)
    elif sample_weighting_method == 'ins':
        weights_for_samples = get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls)
    elif sample_weighting_method == 'isns':
        weights_for_samples = get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls, 0.5)
    else:
        return None
    
    return weights_for_samples


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_class', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--test', type=str2bool, required=True)

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=True)

# dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--border', type=str2bool, required=True)
parser.add_argument('--mixup', type=str2bool, required=True)
parser.add_argument('--label_smooth', type=str2bool, required=True)
parser.add_argument('--se', type=str2bool, required=True)
parser.add_argument('--focal_loss', type=str2bool, required=True)
parser.add_argument('--focal_loss_weight', type=str2bool, required=True)
parser.add_argument('--data_path', type=str, required=True, help='Path to the root directory of the dataset.')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if(args.dataset == 'lrw'):
    from utils import LRWDataset as Dataset
elif(args.dataset == 'lrw1000'):    
    from utils import LRW1000_Dataset as Dataset
elif(args.dataset == 'idlrw'):    
    from utils import IDLRWDataset as Dataset
else:
    raise Exception('lrw or lrw1000 or idlrw')    


video_model = VideoModel(args).cuda()

def parallel_model(model):
    model = nn.DataParallel(model)
    return model        


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
    

lr = args.lr

# adabound
optim_video = optim2.AdaBound(
    video_model.parameters(),
    lr= lr,
    betas= (0.9, 0.999),
    final_lr = 0.1,
    gamma=1e-3,
    eps= 1e-8,
    weight_decay=1e-4,
    # weight_decay=1e-2,
    amsbound=False,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max = args.max_epoch, eta_min=5e-6)


if(args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))    
    load_missing(video_model, weight.get('video_model'))
    
    
video_model = parallel_model(video_model)

def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader =  DataLoader(dataset,
            batch_size = batch_size, 
            num_workers = num_workers,   
            shuffle = shuffle,         
            drop_last = False,
            pin_memory=True,
            generator=g)
    return loader

def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg    

def calculate_class_weights(labels):
    # Count the frequency of each class
    class_counts = np.bincount(labels)
    
    # Calculate the weight for each class
    # The weight is the inverse of the class frequency
    total_count = np.sum(class_counts)
    class_weights = total_count / class_counts
    
    return torch.FloatTensor(class_weights).cuda()

# Path: ARN-Lipreadings/main_visual.py

def val(mode_var=None):
    
    with torch.no_grad():
        if mode_var==None:
            with open('label_sorted.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        elif mode_var=='easy':
            with open('label_sorted_easy.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        elif mode_var=='medium':
            with open('label_sorted_medium.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        elif mode_var=='hard':
            with open('label_sorted_hard.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        args.n_class = len(temp_labels)

        # --- PERBAIKAN: Gunakan 'test' sebagai set validasi ---
        dataset = Dataset('test', args, data_path=args.data_path)
        # --- PERBAIKAN SELESAI ---

        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=True)        
        
        v_acc = []
        val_loss = []
        total = 0
        if (args.focal_loss):
            m = nn.Softmax(dim=-1)

        for (i_iter, input) in enumerate(loader):
            
            video_model.eval()
            
            # Tambahkan pengecekan jika input adalah None (dari dataset_idlrw.py)
            if input is None:
                continue
            
            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            total = total + video.size(0)
            border = input.get('duration').cuda(non_blocking=True).float()
            
            loss2 = {}
            
            if(args.focal_loss):
                var_gamma = 2
                if (args.focal_loss_weight):
                    # Pastikan 'weights' didefinisikan secara global jika digunakan di sini
                    # Untuk keamanan, kita asumsikan tidak pakai weight di validasi
                    loss_fn2 = FocalLoss(gamma=var_gamma)
                else:
                    loss_fn2 = FocalLoss(gamma=var_gamma)
                label_print = "focalloss"
            else:
                if(args.label_smooth):
                    lsr = LSR()
                    loss_fn2 = lsr
                    label_print = "lsr"
                else:
                    loss_fn2 = nn.CrossEntropyLoss()
                    label_print = "CE"
            
            # Hapus with autocast() untuk validasi juga agar konsisten
            if(args.border):
                y_v = video_model(video, border)                                           
            else:
                y_v = video_model(video)                                           
                
            if args.focal_loss:
                loss_bp2 = loss_fn2(m(y_v), label)
            else:
                loss_bp2 = loss_fn2(y_v, label)

            loss2[label_print] = loss_bp2
            val_loss.append(float(loss_bp2.detach().cpu().numpy()))
                         
            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if(i_iter % 10 == 0):  
                msg = ''              
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())                
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)*(len(loader)-i_iter)/3600.0)
                                
        # Handle kasus jika v_acc kosong
        if not v_acc:
             print("Peringatan: Set validasi/test kosong atau semua sampel gagal dimuat.")
             return 0.0, "v_acc=0.00000_", 0.0

        mean_val_loss = np.average(val_loss) if val_loss else 0.0
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)
        
        return acc, msg, mean_val_loss

# Path: ARN-Lipreadings/main_visual.py

# Path: ARN-Lipreadings/main_visual.py

# Path: ARN-Lipreadings/main_visual.py

def test():
    # Menggunakan 'video_model' dan 'args' dari lingkup global
    
    with torch.no_grad():
        dataset = Dataset('test', args, data_path=args.data_path)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)
        
        true_labels, pred_labels, v_acc = [], [], []

        for batch in loader:
            if batch is None: continue
            
            video_model.eval()
            video = batch.get('video').cuda(non_blocking=True)
            label = batch.get('label').cuda(non_blocking=True)
            border = batch.get('duration').cuda(non_blocking=True).float()

            y_v = video_model(video, border) if args.border else video_model(video)
            
            if label is not None:
                true_labels.extend(label.cpu().numpy().tolist())
                pred_labels.extend(y_v.argmax(-1).cpu().numpy().tolist())
                v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())

    if not v_acc:
        print("Tidak ada sampel valid untuk dihitung akurasinya.")
        return 0.0, "v_acc=0.00000_"

    # --- Cetak Hasil Evaluasi ---
    acc = np.mean(v_acc)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    pytorch_total_params = sum(p.numel() for p in video_model.parameters() if p.requires_grad)

    print("-" * 20)
    print("--- Hasil Evaluasi Final ---")
    print(f"Total Sampel Test: {len(true_labels)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total number of parameters: {pytorch_total_params}")
    print("-" * 20)
    
    # --- PERBAIKAN: FLOPs calculation dengan dimensi yang benar ---
    try:
        # Get a sample from the actual dataset to understand the correct input shape
        sample_input = None
        sample_border = None
        
        # Get one sample to understand the actual input dimensions
        for batch in loader:
            if batch is not None:
                sample_input = batch.get('video')[:1]  # Take first sample only
                if args.border:
                    sample_border = batch.get('duration')[:1].float()
                break
        
        if sample_input is not None:
            print(f"Sample input shape: {sample_input.shape}")
            if sample_border is not None:
                print(f"Sample border shape: {sample_border.shape}")
            
            # Create wrapper that matches the actual model interface
            class ModelWrapperForFLOPs(torch.nn.Module):
                def __init__(self, model, use_border=False):
                    super().__init__()
                    self.model = model
                    self.use_border = use_border

                def forward(self, v, border=None):
                    if self.use_border and border is not None:
                        return self.model(v, border)
                    else:
                        return self.model(v)

            model_for_flops_calc = ModelWrapperForFLOPs(
                video_model.module if hasattr(video_model, 'module') else video_model, 
                use_border=args.border
            ).cuda()
            
            # Use the actual sample dimensions for FLOPs calculation
            input_tensor = sample_input.cuda()
            
            if args.border and sample_border is not None:
                border_tensor = sample_border.cuda()
                flops, macs, params = calculate_flops(
                    model=model_for_flops_calc,
                    args=[input_tensor, border_tensor],
                    output_as_string=True,
                    output_precision=4
                )
            else:
                flops, macs, params = calculate_flops(
                    model=model_for_flops_calc,
                    args=[input_tensor],
                    output_as_string=True,
                    output_precision=4
                )
            
            print("Model FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
        else:
            print("Tidak dapat mengambil sample untuk kalkulasi FLOPs")

    except Exception as e:
        print(f"Gagal menghitung FLOPs: {e}")
        print("Mencoba dengan metode alternatif...")
        
        try:
            # Alternative method using FlopCountAnalysis
            # Create a dummy input based on common video input format
            # Assuming video format is [batch, channels, frames, height, width] or similar
            dummy_video = torch.randn(1, 1, 25, 88, 88).cuda()  # Common lipreading format
            
            model_for_analysis = video_model.module if hasattr(video_model, 'module') else video_model
            
            if args.border:
                dummy_border = torch.randn(1, 25).cuda()  # Adjust based on your border format
                flop_analysis = FlopCountAnalysis(model_for_analysis, (dummy_video, dummy_border))
            else:
                flop_analysis = FlopCountAnalysis(model_for_analysis, dummy_video)
            
            total_flops = flop_analysis.total()
            print(f"Model FLOPs (alternative method): {total_flops:,}")
            
        except Exception as e2:
            print(f"Metode alternatif FLOPs juga gagal: {e2}")
            print("Melanjutkan tanpa kalkulasi FLOPs...")
    
    # --- PERBAIKAN SELESAI ---

    return acc, f"v_acc_{acc:.5f}_"       

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)

def train():            

    # # calculate weight for focal loss
    if (args.focal_loss_weight):
        print("Calculating class weights...")
        train_labels = []
        total_batches = len(loader)
        start_time = time.time()
        for i, batch in enumerate(loader):
            labels = batch.get('label')  # Assumes that 'label' is the key for the labels
            train_labels.extend(labels.tolist())

            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (i + 1)) * (total_batches - (i + 1))
            
            print(f"Processed {i + 1}/{total_batches} batches. ETA: {eta:.2f} seconds.")        


        train_labels = np.array(train_labels)
        weights = calculate_class_weights(train_labels)
    else:
        weights = None

    max_epoch = args.max_epoch    
    ce = nn.CrossEntropyLoss()
    tot_iter = 0
    best_before = 0.0
    adjust_lr_count = 0
    alpha = 0.2
    epsilon = 1e-7
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    # scaler = GradScaler() 
    scaler = torch.cuda.amp.GradScaler()          
    mode_list = [None]
    print(len(mode_list))
    patience = 10
    temporary_patience = 0
    chk_epoch = 0

    for mode_var in mode_list:

        lr = args.lr
        optim_video = optim2.AdaBound(
            video_model.parameters(),
            lr= lr,
            betas= (0.9, 0.999),
            final_lr = 0.1,
            gamma=1e-3,
            eps= 1e-8,
            weight_decay=1e-4,
            # weight_decay=1e-2,
            amsbound=False,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max = args.max_epoch, eta_min=5e-6)

        dataset = Dataset('train', args, data_path=args.data_path)
        if mode_var==None:
            with open('label_sorted.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        elif mode_var=='easy':
            with open('label_sorted_easy.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        elif mode_var=='medium':
            with open('label_sorted_medium.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        elif mode_var=='hard':
            with open('label_sorted_hard.txt') as myfile:
                temp_labels = myfile.read().splitlines()
        args.n_class = len(temp_labels)
        
        best_acc = 0.0

        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=True)
        print('Start Training, Data Length:',len(dataset),"Mode:",str(mode_var),"Number of Classes:",str(args.n_class))

        for epoch in range(0, max_epoch):
            total = 0.0
            v_acc = 0.0
            total = 0.0               
            
            lsr = LSR()
            train_acc = []
            train_loss = []
            # softmax for focal loss
            if (args.focal_loss):
                m = nn.Softmax(dim=-1)
                # m = torch.nn.functional.softmax
            for (i_iter, input) in enumerate(loader):
                tic = time.time()           
                
                video_model.train()
                video = input.get('video').cuda(non_blocking=True)
                label = input.get('label').cuda(non_blocking=True).long()  
                if args.border:   
                    border = input.get('duration').cuda(non_blocking=True).float()
                
                loss = {}
                
                if(args.focal_loss):
                    var_gamma = 2
                    if (args.focal_loss_weight):
                        loss_fn = FocalLoss(gamma=var_gamma, weights=weights)
                    else:
                        loss_fn = FocalLoss(gamma=var_gamma)
                    label_print = "focalloss"
                else:
                    if(args.label_smooth):
                        loss_fn = lsr
                        label_print = "lsr"
                    else:
                        loss_fn = nn.CrossEntropyLoss()
                        label_print = "CE"
                # with autocast():
                if(args.mixup):
                    lambda_ = np.random.beta(alpha, alpha)
                    index = torch.randperm(video.size(0)).cuda(non_blocking=True)
                    
                    mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                    if(args.border):
                        mix_border = lambda_ * border + (1 - lambda_) * border[index, :]
                        
                    label_a, label_b = label, label[index]            
                    if(args.border):
                        y_v = video_model(mix_video, mix_border)       
                    else:                
                        y_v = video_model(mix_video)       
                    
                    if args.focal_loss:
                        y_v = m(y_v)
                        y_v = (y_v+epsilon)/(1+100*epsilon)

                    loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)
                    var_acc = (y_v.argmax(-1) == label).cpu().numpy().tolist()
                    var_acc = float(np.array(var_acc).reshape(-1).mean())
                    var_acc = '{:.5f}'.format(var_acc)

                else:
                    if(args.border):
                        y_v = video_model(video, border)       
                    else:                
                        y_v = video_model(video)    
                        
                    if args.focal_loss:
                        loss_bp = loss_fn(m(y_v), label)
                    else:
                        loss_bp = loss_fn(y_v, label)

                    var_acc = (y_v.argmax(-1) == label).cpu().numpy().tolist()
                    var_acc = float(np.array(var_acc).reshape(-1).mean())
                    var_acc = '{:.5f}'.format(var_acc)
                                        
                loss[label_print] = loss_bp
                train_acc.append(float(var_acc))
                train_loss.append(float(loss_bp.detach().cpu().numpy()))

                optim_video.zero_grad()   
                # scaler.scale(loss_bp).backward()  
                # scaler.step(optim_video)
                # scaler.update()

                loss_bp.backward()
                optim_video.step()
                
                toc = time.time()
                
                msg = 'mode={},epoch={},train_iter={},eta={:.5f}'.format(mode_var,epoch, tot_iter, (toc-tic)*(len(loader)-i_iter)/3600.0)
                for k, v in loss.items():                                                
                    msg += ',{}={:.5f}'.format(k, v)
                msg = msg + str(',lr=' + str(showLR(optim_video))) 
                msg = msg + str(',train_acc=' + str(var_acc))                   
                msg = msg + str(',best_val_acc={:2f}'.format(best_acc))
                # print(msg)                                
                
                if(i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0)):

                    acc, msg, mean_val_loss = val(mode_var)
                    writer.add_scalar("Acc/val", acc, tot_iter)
                    mean_train_acc = np.average(train_acc)
                    mean_train_loss = np.average(train_loss)
                    writer.add_scalar("Acc/train", mean_train_acc, tot_iter)
                    writer.add_scalar("Loss/train", mean_train_loss, tot_iter)

                    if(acc > best_acc):
                        # savename = 'best2.pt'
                        # Menggunakan save_prefix dari argumen dan menambahkan nama file
                        # savename = f"{args.save_prefix}_best.pt"
                        savename = f"{args.save_prefix}_best_acc_{acc:.4f}.pt"
                        print(f"Akurasi validasi meningkat. Menyimpan model terbaik ke: {savename}")
                        torch.save(
                            {
                                'video_model': video_model.module.state_dict(),
                            }, savename)         

                    if(tot_iter != 0):
                        best_before = best_acc
                        best_acc = max(acc, best_acc)    
                        
                tot_iter += 1        
            scheduler.step()
            if(acc <= best_before):
                if epoch >= chk_epoch:
                    temporary_patience += 1
                    if(temporary_patience > patience):
                        chk_epoch = epoch 
                        temporary_patience = 0
                        break
            else:
                temporary_patience = 0
            print('epoch={},train_acc={:6f},train_loss={:6f},val_acc={:6f},val_loss={:6f},best_acc={:6f},best_before={:6f},temporary_patience={},chk_epoch={}'.format(epoch, mean_train_acc, mean_train_loss, acc, mean_val_loss, best_acc, best_before, temporary_patience, chk_epoch))
        if chk_epoch == 0:
            chk_epoch = max_epoch

         
        
if(__name__ == '__main__'):
    if(args.test):
        acc, msg = test()
        print(f'acc={acc}')
        writer.add_scalar("Acc/test", acc)
        exit()
    train()
