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


def preprocess(category, save_path, timesteps = 64, data_type='mfcc'):
    category_name = os.path.basename(category)
    
    makedir(f"{save_path}/{category_name}")
    wav_list = glob.glob(os.path.join(category, '*.wav'))
    
    wav2feature = None
    if data_type == 'stft':
        wav2feature = audio.spectrogram
    elif data_type == 'mel':
        wav2feature = audio.melspectrogram
    elif data_type == 'mfcc':
        wav2feature = audio.mfcc
        
    wav2spectrum = lambda x: feature2spectrum(wav2feature(x))
    
    
    for wav_path in wav_list:
        filename = os.path.basename(wav_path)[:-4]
        wav = audio.load_wav(wav_path)

        second = wav.shape[0] // hparams.sample_rate
        window_len = (hparams.sample_rate * second) // (timesteps // 2)
        hop_len = window_len // 2 # overlapping 50%
        
        spectrum_list = []
        for i in range(timesteps):
            if i != timesteps-1:
                window_start = i * hop_len
                window_end = window_start + window_len
            else:
                window_start = -window_len
                window_end   = None

            data = wav[window_start:window_end]
            
            spectrum = wav2spectrum(data)
            spectrum_list.append(spectrum)
            
        sequences = np.stack(spectrum_list, 0) # Shape = (sequence_length, n_dims)
        np.save(f"{save_path}/{category_name}/{filename}.npy", sequences)
    
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', type=int, default=2018, help='-')
    parser.add_argument('--data_type', type=str, default='mel', help='-')
    parser.add_argument('--data_path', type=str, default='../dataset', help='-')
    parser.add_argument('--p', type=float, default='0.25', help='')
    parser.add_argument('--timesteps', type=int, default=64, help='-')
    args, unknown = parser.parse_known_args()
    
    
    """ Split dataset to train and test set"""
    remove_dir = lambda path : shutil.rmtree(path, ignore_errors=True) if os.path.exists(path) else None
    trainset_path = os.path.join(args.data_path, f'{args.data_version}_train')
    testset_path  = os.path.join(args.data_path, f'{args.data_version}_test')
    remove_dir(trainset_path)
    os.makedirs(trainset_path, exist_ok=True)
    remove_dir(testset_path)
    os.makedirs(testset_path, exist_ok=True)
    
    dataset_list = glob.glob(os.path.join(args.data_path, f'{args.data_version}_set', '*'))
    for category_path in dataset_list:
        category_name = os.path.basename(category_path)
        makedir(os.path.join(trainset_path, category_name))
        makedir(os.path.join(testset_path, category_name))
        fileslist = glob.glob(os.path.join(category_path, '*'))
        
        random.seed(2019)
        random.shuffle(fileslist)
        
        n_files = len(fileslist)

        #if n_files <= 15 : args.p = 0.4 # if a category a small amount of data, increase the amount of test data.
            
        files4train = fileslist[:int(n_files * (1-args.p))]
        files4test  = fileslist[int(n_files *(1-args.p)):]
        
        for src in files4train:
            filename = os.path.basename(src)
            dst = os.path.join(trainset_path, category_name, filename)
            
            shutil.copy(src, dst) if not os.path.exists(dst) else None
            
        for src in files4test:
            filename = os.path.basename(src)
            dst = os.path.join(testset_path, category_name, filename)
            
            shutil.copy(src, dst) if not os.path.exists(dst) else None
        
        
    """ Preprocessing """
    trainsetPath_list = glob.glob(os.path.join(trainset_path, '*'))
    testsetPath_list = glob.glob(os.path.join(testset_path, '*'))
    
    
    n_dims = None
    if args.data_type == 'stft':
        n_dims = int(1 + hparams.n_fft/2)
    elif args.data_type == 'mel':
        n_dims = hparams.n_mels
    elif args.data_type == 'mfcc':
        n_dims = hparams.n_mfcc
    
    save_path = f"../data_{args.data_type}_{n_dims}_dims"
    save_train_path = os.path.join(save_path, f'{args.data_version}_train')
    save_test_path = os.path.join(save_path, f'{args.data_version}_test')
    os.makedirs(save_train_path, exist_ok=True)
    os.makedirs(save_test_path, exist_ok=True)

    print("Preprocessing train set")
    for category_path in trainsetPath_list:
        preprocess(category_path, save_train_path, timesteps=args.timesteps, data_type=args.data_type)
    remove_dir(trainset_path)
    print("Done")
        
    print("Preprocessing test set")
    for category_path in testsetPath_list:
        preprocess(category_path, save_test_path, timesteps=args.timesteps, data_type=args.data_type)
    remove_dir(testset_path)
    print("Done")
    
    
