import numpy as np
import os 
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sources import utils_spec, loss, models, MMFusion_Model
import matplotlib.pyplot as plt


import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from tensorflow.signal import stft
from torchmetrics.classification import Accuracy


from train_lib import *



def main():
    
    batch_size = 32
    epochs = 30
    
    save_models_to = os.path.join(path, "models")
    os.makedirs(save_models_to)

    save_history_to = os.path.join(path, "history")
    os.makedirs(save_history_to)

    save_plots_to = os.path.join(path, "plots")
    os.makedirs(save_plots_to)


    for window_size in np.arange(128, 256, 8):
        for overlap in np.arange(32, window_size, 32):
            
            train_loader, test_loader = get_data(num_classes, "pretraining", batch_size, window_size, overlap)
            
            print_log("Experiment config : window_size {}, overlap {}".format(window_size, overlap), log)

            data_iter = iter(train_loader)
            images, *_ = next(data_iter)
            print_log("=" * 150, log)
            print_log("Contrastive Learning step\n", log)

            try:
                _, _, fusion_model = get_models(images.shape[1:])
            except:
                print_log("Skipping The Experiment", log)
                print_log("window_size {}, overlap {} Not Adapted for Training".format(window_size, overlap), log)
                print_log("=" * 150, log)
                continue
        
            save_to = os.path.join(save_models_to, 'ConvRNN_Model_window_size{}_step{}.pth'.format(window_size, overlap))
            
            fusion_model, contrastive_history = contrastive_learning(model=fusion_model, train_loader=train_loader, val_loader=test_loader, batch_size = batch_size, loss=loss, epochs=epochs, checkpoint=save_to)
                
            np.save(os.path.join(save_history_to, "contrastive_history_windowsize{}_overlap{}.npy".format(window_size, overlap)), contrastive_history)

            print_log("", log)
            print_log("=" * 150, log)
            print_log("FineTuning step\n", log)

            train_loader, test_loader = get_data(num_classes, "finetuning", batch_size, window_size, overlap)

            ft_model = models.downstream_model.DSModel_Fusion(fusion_model, num_classes=num_classes, fusion=True).to(device)

            save_to = os.path.join(save_models_to, 'ConvRNN_finetuned_TF_window_size{}_overlap{}.pth'.format(window_size, overlap))

            ft_model, finetune_history = finetune(model=ft_model, train_loader=train_loader, val_loader=test_loader, epochs=epochs, checkpoint=save_to)

            np.save(os.path.join(save_history_to, "finetune_history_windowsize{}_overlap{}.npy".format(window_size, overlap)), finetune_history)

            
            plot_values(contrastive_history, title = "Contrastive_window_size {}, overlap {}".format(window_size, overlap), save_to = save_plots_to)
            plot_values(finetune_history, title = "FineTuning_window_size {}, overlap {}".format(window_size, overlap), plot_accuracy = True, save_to = save_plots_to)


            print_log("", log)
            print_log("=" * 150, log)
            print_log("Evaluation step\n", log)

            # ========================================================================================================================================================================
                                                    # EVALUATION WORKS ONLY FOR BINARY CLASS !!!!!!!!!!!!!!!!!!!!!!!!!
                                                    # UNCOMMENT THE FOLLOWING CODE FOR EVALUATION
            # ========================================================================================================================================================================

            # data_loader = get_data(num_classes=num_classes, mode = "evaluation", batch_size=batch_size, window_size=window_size, overlap=overlap, split = False)

            # save_to = os.path.join(save_plots_to, "ConfusionMatrix_window_size {}, overlap {}".format(window_size, overlap))
            # fpr_nn, tpr_nn, th_nn = evaluate(ft_model, data_loader, save_to)
            
            # plt.figure(figsize = (8, 4))
            # plt.plot(fpr_nn, tpr_nn)
            # plt.title("ROC")
            # plt.ylabel("tpr_nn")
            # plt.xlabel("fpr_nn")
            # plt.savefig(os.path.join(save_plots_to, "metrics_window_size {}, overlap {}".format(window_size, overlap)))
        
            # auc_model = auc(fpr_nn, tpr_nn)

            print_log("End of Experiment", log)
            print_log("=" * 150, log)
            print_log("", log)


if __name__ == "__main__":
    main()