import os
import matplotlib.pyplot as plt
import numpy as np

def save_loss(loss_list, model_path):
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(loss_list, color='blue', linestyle="-", label="loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(f"{model_path}/loss.png")
    plt.close()
    
    
def compare_loss(loss_dict, model_path, value, save_figure_path, weight=1.2, log_scaling=0):
    plt.clf()
    plt.rcParams['font.size'] = 15
    
    xlabel = "index of data"
    ylabel = "loss -- y=log10(x+10)" if log_scaling==1 else "loss"
    
    for key, loss_list in loss_dict.items():
        if log_scaling == 1:
            loss_list = np.log10(np.array(loss_list) + 10)
        
        plt.scatter(range(len(loss_list)), loss_list, label=key, s=10)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if value != None:
        threshold = value * weight
        plt.axhline(y=threshold, color='r', linestyle='-')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size' : 12})
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    plt.savefig(f"{model_path}/compare_loss.png")
    plt.savefig(f"{save_figure_path}_compare_loss.png")
    
    plt.ylim(0, value * 5)
    plt.savefig(f"{model_path}/compare_loss_zoom.png")
    plt.savefig(f"{save_figure_path}_compare_loss_zoom.png")

    plt.close()
