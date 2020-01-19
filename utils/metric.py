import numpy as np


def get_score(loss_dict, target_name, value, timesteps, inputs_dims, weight):
    score_dict = {}
    TPR = None # Recall
    FPR = None
    recall = None
    precision = None
    TP = 0 # True Positive
    FN = 0 # False Negative
    FP = 0 # False Positive
    Ne = 0 # Negative
    eps = 1e-8

    threshold = value * weight
            
    for name, loss_list in loss_dict.items():
        if name in target_name: # Real True
            for loss_val in loss_list:
                if loss_val <= threshold: # Predict : True
                    TP += 1
                elif loss_val > threshold: # Predict : False
                    FN += 1
        else: # Real False 
            Ne += len(loss_list)
            for loss_val in loss_list:
                if loss_val <= threshold: # Predict : True
                    FP += 1
    
    TPR = recall = TP / (TP + FN + eps) 
    FPR = FP / Ne
    precision = TP / (TP + FP + eps) 
    
    F1_SCORE = 2 * (precision * recall ) / (precision + recall + eps)
    
    
    score_dict = {
        'name' : target_name,
        'weight' : weight,
        'seq_length' : timesteps,
        'dims' : inputs_dims,
        'TPR' : round(TPR, 3),
        'FPR' : round(FPR, 3),
        'F1_SCORE' : round(F1_SCORE, 3)
    }
    
    return score_dict
