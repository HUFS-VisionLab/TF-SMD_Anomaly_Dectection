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


def preprocess(category, save_path, timesteps = 64, type='mfcc'):
    category_name = os.path.basename(category)
    
    makedir(f"{save_path}/{category_name}")
    wav_list = glob.glob(os.path.join(category, '*.wav'))
    
    wav2feature = None
    if type == 'stft':
        wav2feature = audio.spectrogram
    elif type == 'mel':
        wav2feature = audio.melspectrogram
    elif type == 'mfcc':
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
    parser.add_argument('--data_type', type=str, default='mel', help='-')
    parser.add_argument('--data_path', type=str, default='../dataset', help='-')
    parser.add_argument('--p', type=float, default='0.25', help='')
    parser.add_argument('--timesteps', type=int, default=64, help='-')
    args, unknown = parser.parse_known_args()
    
    
    """ Split dataset to train and test set"""
    remove_dir = lambda path : shutil.rmtree(path, ignore_errors=True) if os.path.exists(path) else None
    trainset_path = os.path.join(args.data_path, 'train')
    testset_path  = os.path.join(args.data_path, 'test')
    remove_dir(trainset_path)
    makedir(trainset_path)
    remove_dir(testset_path)
    makedir(testset_path)
    
    dataset_list = glob.glob(os.path.join(args.data_path, 'data', '*'))
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
    trainset_path = glob.glob(os.path.join(trainset_path, '*'))
    testset_path = glob.glob(os.path.join(testset_path, '*'))
    
    save_path = f"../data_{args.data_type}"
    save_train_path = os.path.join(save_path, 'train')
    save_test_path = os.path.join(save_path, 'test')
    makedir(save_path)
    makedir(save_train_path)
    makedir(save_test_path)
    
    print("Preprocessing train set")
    for category_path in trainset_path:
        preprocess(category_path, save_train_path, timesteps=args.timesteps, type=args.data_type)
    print("Done")
        
    print("Preprocessing test set")
    for category_path in testset_path:
        preprocess(category_path, save_test_path, timesteps=args.timesteps, type=args.data_type)
    print("Done")
