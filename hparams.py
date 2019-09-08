# Hyperparameters for preprocessing
class hparams:
    # Audio
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
