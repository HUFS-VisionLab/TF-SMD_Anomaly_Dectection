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
- `hparams.py` : hyper parameter for preprocessing audio data.
- `utils/audio.py` : audio preprocessing functions.
- `utils/preprocessing.py` : main preprocessing code.
- `model.py` : nn model code.
- `plot.py` : code to visualize results.
- `run.py` : main code to execute training and test.
- `dataset` : dataset.

## How to use
#### Hyper Parameter for preprocessing
```
class hparams:
    # for audio data
    sample_rate = 192 * 1000
    preemphasis = 0.97
    n_fft = 2048
    win_length = 2048
    hop_length = 2048 // 4
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
- `--data_type`: Type of training data. *Choices*: `'mel'`, `'mfcc'`, `'stft'`. *Default*: `'mel'`
- `--data_path`: Path of the dataset. *Default*: `../dataset`
- `--p`: Ratio of test set. *Default*: `0.25`
- `--timesteps`: Length of input sequence. *Default*: `64`
#### Running
```
python run.py
```

*Required* :
- `--target_category` : List of category to train. *Default*: `["AT"]`. *Ex*: `--target_category AT ST` -> ["AT, ST"]

*Optional* :  
- `--n_layers`: The number of layers. *Default*: `3`
- `--is_bidirectional`: *Choices*: 0(Unidirectional), 1(Bidirectional). *Default*: `1`
- `--learning_rate`: The value of learning rate for Adam Optimizer. *Default*: `0.005`
- `--beta_1`: Beta_1 of Adam Optimizer. *Default*: `0.9`
- `--beta_2`: Beta_2 of Adam Optimizer. *Default*: `0.999`
- `--epsilon`: Epsilon of Adam Optimizer. *Default*: `1e-08`
- `--epochs`: The number of epochs. *Default*: `1000`
- `--batch_size`: The size of mini-batch. *Default*: `None`. *type*: int
- `--data_path`: Path of the dataset. *Default*: `../dataset`
- `--model_path`: Save path of trained model. *Default*: `./model`
- `--inference`: *Choices*: False(train), True(test). *Default*: `False`
