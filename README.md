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
- `hparams.py` : hyper parameter for audio file of user. IMPORTANT
- `utils/preprocessing.py` : preprocessing code.
- `utils/metric.py` : metric functions.
- `ssae.py` : SSAE model code.
- `plot.py` : code to visualize results.
- `run.py` : main code to execute training and test.
- `benchmark.py` : code to benchmark.

## How to use
#### Hyper Parameter for preprocessing
```
class hparams:
    # Hyper Parameters for Audio 
    sample_rate = 192 * 1000 # Based on SMD audio info
    window_length = 0.02 #20ms
    window_stride = 0.005 # 5ms
    n_fft = int(sample_rate * window_length) 
    win_length = n_fft
    hop_length = int(sample_rate * window_stride) 
```

#### Preprocessing
```
cd ./utils
python preprocessing
```

*Optional* :  
- `--dataset-path`: Path of the dataset. *Default*: `../dataset`
- `--dataset-name` : Name of the dataset. *choices*: 'SMD_dataset', 'Custom_dataset', ...
- `--n-mels` : the number of mel filters or frequency dimensions. *Default*: `80`
- `--seq-len` : the value of target length of output sequence. Hyperparameter for *Sequence Normalize*. *Default*: `32`

#### Running
```
python run.py

ex)
python run.py --n_layers 4 --epochs 5000 --dataset-name SMD_dataset --seq-len 32 --dims 80 --targets CLR-085
```

*Required* :
- `--targets` : List of category to train. *Default*: `None`.
  *Ex*: `--targets AT` -> ["AT"], `--targets AT ST` -> ["AT, ST"] 
- `--n_layers`: The number of layers. *Default*: `4`
- `--epochs`: The number of epochs. *Default*: `5000`

*Optional* :  
- `--dataset-name`: The name of target dataset. *Default*: `SMD_dataset`
- `--seq-len`: The length of input sequence. *Default*: `32`
- `--dims`: The frequency dimension of input sequence. *Default*: `80`
- `--no_bidirectional`: action='store_true', *Default*: `False`
- `--learning_rate`: The value of learning rate for Adam Optimizer. *Default*: `0.005`
- `--beta_1`: Beta_1 of Adam Optimizer. *Default*: `0.9`
- `--beta_2`: Beta_2 of Adam Optimizer. *Default*: `0.999`
- `--epsilon`: Epsilon of Adam Optimizer. *Default*: `1e-08`
- `--batch_size`: The size of mini-batch. *Default*: `None`. *type*: int
- `--model_path`: Save path of trained model. *Default*: `./model`
- `--test`: *Default*: `False`
- `--threshold_weight`: the value of weight to make threshold. *Default*: `2.0`
