import os
import time


epochs_list = [5000]
n_layers_list = [5]
model_type_list = [2]
data_type = 'timesteps_64_mel_80'

targets_2018 = [
                   "AT", "M1", "M2", "NA1", "NA2", "ST",
                   "AT NA1", "AT NA2", "AT ST", "NA1 NA2", "AT ST", "NA1 M1", "NA2 M2", "M1 M2", "M1 ST", "M2 ST",
                   "AT NA1 NA2", "AT NA1 M1", "AT NA2 M2", "NA1 NA2 ST", "NA1 M1 ST", "NA2 M2 ST", "AT M1 ST", "AT M1 M2",
                   "M1 M2 ST"
               ]

targets_2019 = [
                   "CLR", "MG", "ST1", "ST2", "ST3", "TSIO", "NW"
               ]


for i in range(0, len(epochs_list)):
    epochs = epochs_list[i]
    n_layers = n_layers_list[i]
    model_type = model_type_list[i]
    start = time.time()
    print(f"Start {i+1} simulate")
    for target_category in targets_2019:
        os.system(f'python run.py --epochs {epochs} --n_layers {n_layers} --model_type {model_type} --data_type {data_type} 
                  --target_category {target_category}')
        
    end = time.time()
    print(f"Done. Elapsed time: {end-start}")
    
    
print("Clear")
