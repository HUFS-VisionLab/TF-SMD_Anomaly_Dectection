import os
import time


epochs = 5000
n_layers = 4
model_type_list = [0] #[0, 1, 2, 3, 4]
data_type = 'timesteps_64_mel_80'

targets_2018 = [
                   "AT", "M1", "M2", "NA1", "NA2", "ST",
                   "AT NA1", "AT NA2", "AT ST", "NA1 NA2", "AT ST", "NA1 M1", "NA2 M2", "M1 M2", "M1 ST", "M2 ST",
                   "AT NA1 NA2", "AT NA1 M1", "AT NA2 M2", "NA1 NA2 ST", "NA1 M1 ST", "NA2 M2 ST", "AT M1 ST", "AT M1 M2",
                   "M1 M2 ST"
               ]

targets_2019_1 = [
                   "CLR", "MG", "ST1", "ST2", "ST3", "TSIO", "NW"
                  ]

targets_2019_2 = [
                   "CLR-2", "MG-2", "MA", "NA-2"
                  ]

"""
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
    """
                  
                  
for targets in targets_2019_2:
    for model_type in model_type_list:
        command = f'python run.py --epochs {epochs} --n_layers {n_layers} --model_type {model_type} --data_type {data_type} --targets {targets}'
        os.system(command)
        os.system(f'{command} --inference')
        
        command = f'python run.py --epochs {epochs} --n_layers {n_layers} --model_type {model_type} --data_type {data_type} --targets {targets} --augment'
        os.system(command)
        os.system(f'{command} --inference')
    
print("Clear")