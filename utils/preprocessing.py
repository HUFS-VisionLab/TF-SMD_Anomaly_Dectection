import os
import sys 
import glob
import scipy
import shutil
import random
import librosa
import numpy as np
import argparse


sys.path.append(os.path.join(os.getcwd(), '..'))
from hparams import hparams


def min_max_scaling(x):
    """
    Args:
        S: np.array, Spectrogram. Shape=(f, t)
    Returns:
        S: np.array, scalied Spectrogram. Shape=(f, t)
    """
    _max = np.max(x)
    _min = np.min(x)
    
    x = (x - _min + 1e-7)  / (_max - _min)
    
    return x
        
        
def time_wise_average(S):
    """ Summation or Average by time
    Args:
        S : np.array, Spectrogram. Shape=(n_mfcc, time) or (frame_bins, time)
        
    Returns:
        spect : np.array, spectrum. Shape=(n_mfcc) or (frame_bins)
    """
    spectrum = np.mean(S, axis=-1)

    return spectrum


def preprocess(target_dir, save_path, args):
    target_name = os.path.basename(target_dir)
    print(f">> target : {target_name}")
    for saveDir_name in ['train', 'test', 'train_shifted', 'test_shifted']:
        saveDir_path = os.path.join(save_path, target_name, saveDir_name)
        os.makedirs(saveDir_path, exist_ok=True)

    join_path = lambda x,y : os.path.join(x,y)
    trainPath_list = glob.glob(join_path(target_dir, 'train/*.wav'))
    testPath_list = glob.glob(join_path(target_dir, 'test/*.wav'))
    dataPath_dict = {'train' : trainPath_list, 'test' : testPath_list}
    
    for key, path_list in dataPath_dict.items():
        if len(path_list) % 2 != 0:
            path_list += [path_list[0]]
            
        for i in range(int(len(path_list)/2)):
            idx = i*2
            path_1 = path_list[idx] 
            path_2 = path_list[idx+1]
            
            
            # Original
            name_1 = os.path.basename(path_1)[:-4]
            name_2 = os.path.basename(path_2)[:-4]
            wav_1 = librosa.load(path_1, sr=hparams.sample_rate)[0]
            wav_2 = librosa.load(path_2, sr=hparams.sample_rate)[0]
            
            
            # Time shift (Augment)
            timeshift = 0.5 # 500ms
            new_wav = np.concatenate([wav_1, wav_2])
            
            start = int(hparams.sample_rate*timeshift) # 0.5 is 500ms
            end = start + wav_1.shape[0]
            wav_3 = new_wav[start:end]
            
            start = wav_1.shape[0] - int(hparams.sample_rate*timeshift)
            end = - int(hparams.sample_rate*timeshift)
            wav_4 = new_wav[start:end]
            
            
            wav_list = [wav_1, wav_2, wav_3, wav_4]
            name_list = [name_1, name_2, f'{name_1}_shifted', f'{name_2}_shifted']
            
            for j in range(len(wav_list)):
                wav = wav_list[j]
                name = name_list[j]
                
                S = librosa.feature.melspectrogram(y=wav, sr=hparams.sample_rate,
                                                   n_fft=hparams.n_fft,
                                                   win_length=hparams.win_length,
                                                   hop_length=hparams.hop_length,
                                                   n_mels=args.n_mels)
                S_len = S.shape[1]
                
                # Sequence Normalize
                over_lapping = 0.5 # 0 <= over-lapping < 1 
                window_length = S_len // int(args.seq_len * (1-over_lapping))
                stride = int(window_length * (1-over_lapping))
                
                spectrum_list = []
                for i in range(args.seq_len):
                    if i != args.seq_len-1:
                        window_start = i * stride
                        window_end = window_start + window_length
                    else:
                        window_start = - window_length
                        window_end = None

                    local_S = S[:,window_start:window_end]
                    spectrum = time_wise_average(local_S)
                    spectrum_list.append(spectrum)

                sequence = np.stack(spectrum_list, 0) # Shape = (sequence_length, n_dims)
                sequence = min_max_scaling(sequence)
                
                if j < 2:
                    np.save(f"{save_path}/{target_name}/{key}/{name}.npy", sequence)
                else:
                    np.save(f"{save_path}/{target_name}/{key}_shifted/{name}.npy", sequence)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', dest='dataset_path', type=str, default='../dataset', help='-')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str, default='SMD_dataset', help='-')
    parser.add_argument('--n-mels', dest='n_mels', type=int, default=80, help='-')
    parser.add_argument('--seq-len', dest='seq_len', type=int, default=32, help='-')
    args, unknown = parser.parse_known_args()
    
    targetDir_list = glob.glob(os.path.join(args.dataset_path, args.dataset_name, '*'))
    targetDir_list = sorted(targetDir_list)
        
    save_path = "../seqlen_{}_mels_{}/{}".format(args.seq_len, args.n_mels, args.dataset_name)

    print(">> Preproecssing..")
    for target_dir in targetDir_list:
        preprocess(target_dir, save_path, args)
    print(">> Done")   
