class hparams:
    # Hyper Parameters for SMD audio file 
    sample_rate = 192 * 1000
    preemphasis = 0.97
    window_length = 0.01 #10ms
    window_stride = window_length / 4 # 5ms, 25% over-lapping
    n_fft = int(sample_rate * window_length) 
    win_length = n_fft
    hop_length = int(sample_rate * window_stride) 
