# TF-SMD_Anomaly_Dectection

## Environment
- Ubuntu 16.04
- Python 3.7

## Depenency
- Numpy
- Opencv2
- Matplotlib
- librosa
- scipy
- Tensorflow (1.10 <= x <=1.14)

## Files
- `dataset/` : dataset directory
- `hparams.py` : hyper parameter for preprocessing audio data. IMPORTANT
- `utils/audio.py` : audio preprocessing functions.
- `utils/preprocessing.py` : main preprocessing code.
- `model.py` : nn model code.
- `tf_models.py` : tf modules for nn model code.
- `plot.py` : code to visualize results.
- `run.py` : main code to execute training and test.
- `benchmark.py` : code to benchmark.
- `simulate.py` : script to run main process.

## How to use
#### Hyper Parameter for preprocessing
```
class hparams:
    # Hyper Parameter for preprocessing
    data_type = 'mel'
    seq_length = 64
    timeshift = 0.5 # 500ms

    # Hyper Parameters for Audio 
    sample_rate = 192 * 1000
    preemphasis = 0.97
    window_length = 0.02 #20ms
    window_stride = 0.01 #10ms
    n_fft = int(sample_rate * window_length) 
    win_length = n_fft
    hop_length = int(sample_rate * window_stride) 
    n_mels = 80
    n_mfcc = 60
    min_level_db = -100
    ref_level_db = 20
    max_iters=200
    griffin_lim_iters=60
    power=1.5  
```

#### Preprocessing
```
cd ./utils
python preprocessing
```

*Optional* :  
- `--data_version` : Version of dataset. *choices*: '2018', '2019-1', '2019-2'
- `--data_path`: Path of the dataset. *Default*: `../dataset`

#### Running
```
python run.py

ex)
python run.py --epochs 5000 --n_layers 4 --model_type 0 --data_type timesteps_64_mel_80 --targets CLR
```

*Required* :
- `--targets` : List of category to train. *Default*: `None`.
  *Ex*: `--targets AT` -> ["AT"], `--targets AT ST` -> ["AT, ST"] 

*Optional* :  
- `--n_layers`: The number of layers. *Default*: `3`
- `--mode_type`: *Choices*: 0(Basic), 1(Autoencoder), 2(AutoEncoder_context), 3(OneClass), 4(OneClass_condition). *Default*: `2`
- `--no_bidirectional`: action='store_true', *Default*: `False`
- `--data_type`: The detail of feature extraction. *Default*: `timesteps_64_mel_80`
- `--learning_rate`: The value of learning rate for Adam Optimizer. *Default*: `0.005`
- `--beta_1`: Beta_1 of Adam Optimizer. *Default*: `0.9`
- `--beta_2`: Beta_2 of Adam Optimizer. *Default*: `0.999`
- `--epsilon`: Epsilon of Adam Optimizer. *Default*: `1e-08`
- `--epochs`: The number of epochs. *Default*: `3000`
- `--batch_size`: The size of mini-batch. *Default*: `None`. *type*: int
- `--model_path`: Save path of trained model. *Default*: `./model`
- `--inference`: *Choices*: False(train), True(test). *Default*: `False`
- `--loss_log_scale`: If you want to see log scaled loss value. *Choice*: `0(False)`, `1(True)`.
