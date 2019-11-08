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