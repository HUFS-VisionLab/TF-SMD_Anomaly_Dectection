import os
import sys 
import glob
import shutil
import random
import librosa
import numpy as np
import argparse

import audio

sys.path.append(os.path.join(os.getcwd(), '..'))
from hparams import hparams


def makedir(path):
    try:
        os.mkdir(path)
        print("Make dir: %s" %(path))
    except:
        print("Already exist: %s" %(path))
        
        
def normalize(S):
    """
    Args:
        S: np.array, Spectrogram. Shape=(f, t)
    Returns:
        S: np.array, normalized Spectrogram. Shape=(f, t)
    """
    mean = np.expand_dims(np.mean(S, axis=1), axis=1)
    std = np.expand_dims(np.std(S, axis=1), axis=1)
    
    S = (S - mean)/std
    
    return S
        
        
def feature2spectrum(S, mode='avg'):
    """ Summation or Average by time
    Args:
        S : np.array, normalized spectrogram. Shape=(n_mfcc, time) or (frame_bins, time)
        
    Returns:
        spect : np.array, spectrum. Shape=(n_mfcc) or (frame_bins)
    """
    func = None
    if mode == 'avg':
        func = np.mean
    if mode == 'sum':
        func = np.sum
        
    spect = func(S, axis=-1)

    return spect


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
    
    wav2feature = set_dict[hparams.data_type]['func']
    wav2spectrum = lambda x: feature2spectrum(wav2feature(x))
    
    
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
            wav_1 = audio.load_wav(path_1)
            wav_2 = audio.load_wav(path_2)
            
            # Time shift (Augment)
            new_wav = np.concatenate([wav_1, wav_2])
            
            start = int(hparams.sample_rate*hparams.timeshift)
            end = start + wav_1.shape[0]
            wav_3 = new_wav[start:end]
            
            start = wav_1.shape[0] - int(hparams.sample_rate*hparams.timeshift)
            end = - int(hparams.sample_rate*hparams.timeshift)
            wav_4 = new_wav[start:end]
            
            
            wav_list = [wav_1, wav_2, wav_3, wav_4]
            name_list = [name_1, name_2, f'{name_1}_shifted', f'{name_2}_shifted']
            
            for j in range(len(wav_list)):
                wav = wav_list[j]
                name = name_list[j]
                
                second = wav.shape[0] // hparams.sample_rate
                window_len = (hparams.sample_rate * second) // (hparams.seq_length // 2)
                hop_len = window_len // 2 # overlapping 50%
                
                spectrum_list = []
                for i in range(hparams.seq_length):
                    if i != hparams.seq_length-1:
                        window_start = i * hop_len
                        window_end = window_start + window_len
                    else:
                        window_start = -window_len
                        window_end   = None

                    data = wav[window_start:window_end]

                    spectrum = wav2spectrum(data)
                    spectrum_list.append(spectrum)

                sequences = np.stack(spectrum_list, 0) # Shape = (sequence_length, n_dims)
                
                if j < 2:
                    np.save(f"{save_path}/{target_name}/{key}/{name}.npy", sequences)
                else:
                    np.save(f"{save_path}/{target_name}/{key}_shifted/{name}.npy", sequences)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', type=int, default=2019, help='-')
    parser.add_argument('--data_path', type=str, default='../dataset', help='-')
    args, unknown = parser.parse_known_args()
    
    
    targetDir_list = glob.glob(os.path.join(args.data_path, f'{args.data_version}_set', '*'))
        
    
    set_dict={
        'stft' : {
            'func' : audio.spectrogram,
            'n_dims' : int(1 + hparams.window_length/2)
        },
        'mel' : {
            'func' : audio.melspectrogram,
            'n_dims' :  hparams.n_mels
        },
        'mfcc' : {
            'func' : audio.mfcc,
            'n_dims' :  hparams.n_mfcc
        }
    }
    
    data_type = hparams.data_type
    seq_length = hparams.seq_length
    n_dims = set_dict[data_type]['n_dims']
    
    save_path = f"../timesteps_{seq_length}_{data_type}_{n_dims}"

    print(">> Preproecssing..")
    for target_dir in targetDir_list:
        preprocess(target_dir, save_path, set_dict)
    print(">> Done")   