#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from itertools import chain
from collections import namedtuple
import pickle
import os.path
import shutil
import inspect 
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data

from pytorch_utils import *
from pas_utils import *

if __name__=="__main__":
    MODEL_DIR="./pytorch_models"
    PROCESSED_DIR="./APA_ML/processed"
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    params={
        "batch_size":32,
        "lr":1e-3,
        "beta":1e-3,
        "net_type":"Multi-Conv-Net",    
        "conv1d_kernel_size":12,
        
        "conv1d_out_dim_1":40,
        "pool_size_1":3,
        "conv1d_out_dim_2":40,
        "pool_size_2":4,
        
        "linear1_dim":200,
        "seq_len":455,
        "lstm_output_size":100,
        "device":"cuda" if torch.cuda.is_available() else "cpu",
        "parental_model_file":os.path.join(MODEL_DIR,"parental_model_bl-multi.pt"),
        "f1_model_file":os.path.join(MODEL_DIR,"f1_model_from_bl-multi.pt"),
        "dropout_rate":0.7,
        "fold":5
    }
    if os.path.exists(params["f1_model_file"]):
        raise FileExistsError("%s already exists"%(params["f1_model_file"]))
    
    PARENTAL_SEQUENCE_TABLE_FILE=os.path.join(PROCESSED_DIR,'parental_sequence_table.h5')
    PARENTAL_BL_PROCESSED_SEQUENCE_FILE=os.path.join(PROCESSED_DIR,'processed_parental_bl_sequences.npy')
    PARENTAL_SP_PROCESSED_SEQUENCE_FILE=os.path.join(PROCESSED_DIR,'processed_parental_sp_sequences.npy')
    PARENTAL_BL_PROCESSED_FEATURES_FILE=os.path.join(PROCESSED_DIR,'processed_parental_bl_features.npy')
    PARENTAL_SP_PROCESSED_FEATURES_FILE=os.path.join(PROCESSED_DIR,'processed_parental_sp_features.npy')
    F1_SEQUENCE_TABLE_FILE=os.path.join(PROCESSED_DIR,'f1_sequence_table.h5')
    F1_BL_PROCESSED_SEQUENCE_FILE=os.path.join(PROCESSED_DIR,'processed_f1_bl_sequences.npy')
    F1_SP_PROCESSED_SEQUENCE_FILE=os.path.join(PROCESSED_DIR,'processed_f1_sp_sequences.npy')
    F1_BL_PROCESSED_FEATURES_FILE=os.path.join(PROCESSED_DIR,'processed_f1_bl_features.npy')
    F1_SP_PROCESSED_FEATURES_FILE=os.path.join(PROCESSED_DIR,'processed_f1_sp_features.npy')
    parental_sequence_table=pd.read_hdf(PARENTAL_SEQUENCE_TABLE_FILE)
    parental_bl_processed_sequences=np.load(PARENTAL_BL_PROCESSED_SEQUENCE_FILE)
    parental_sp_processed_sequences=np.load(PARENTAL_SP_PROCESSED_SEQUENCE_FILE)
    parental_gene_ids=list(np.load(os.path.join(PROCESSED_DIR,'parental_gene_ids_shuffled.npy')))
    parental_bl_processed_features=np.load(PARENTAL_BL_PROCESSED_FEATURES_FILE)
    parental_sp_processed_features=np.load(PARENTAL_SP_PROCESSED_FEATURES_FILE)
    f1_sequence_table=pd.read_hdf(F1_SEQUENCE_TABLE_FILE)
    f1_bl_processed_sequences=np.load(F1_BL_PROCESSED_SEQUENCE_FILE)
    f1_sp_processed_sequences=np.load(F1_SP_PROCESSED_SEQUENCE_FILE)
    f1_gene_ids=list(np.load(os.path.join(PROCESSED_DIR,'f1_gene_ids_shuffled.npy')))
    f1_bl_processed_features=np.load(F1_BL_PROCESSED_FEATURES_FILE)
    f1_sp_processed_features=np.load(F1_SP_PROCESSED_FEATURES_FILE)
    parental_processed_sequences=np.concatenate([parental_bl_processed_sequences,parental_sp_processed_sequences],axis=0)

    with open(os.path.join(PROCESSED_DIR,'data_dict_folds-parental-bl-sequences-455nt.pkl'),'rb') as f:
        parental_bl_data_dict_folds=pickle.load(f)
    with open(os.path.join(PROCESSED_DIR,'data_dict_folds-f1-bl-sequences-455nt.pkl'),'rb') as f:
        f1_bl_data_dict_folds=pickle.load(f)   
    with open(os.path.join(PROCESSED_DIR,'data_dict_folds-f1-sp-sequences-455nt.pkl'),'rb') as f:
        f1_sp_data_dict_folds=pickle.load(f) 
    f1_combined_data_dict_folds=blsp_folds_combine(f1_bl_data_dict_folds,f1_sp_data_dict_folds)

    parental_fold_data_dict={}
    fold_train=fold_combine([parental_bl_data_dict_folds[(params["fold"]+i)%5] for i in [0,1,2,3]],'train')
    fold_test=fold_combine([parental_bl_data_dict_folds[(params["fold"]+i)%5] for i in [4]],'dev')
    parental_fold_data_dict.update(fold_train)
    parental_fold_data_dict.update(fold_test)
    # TODO: make dataset that contains pas_numbers_comparison and remove these two assertions
    assert np.allclose(parental_fold_data_dict["X_train"],parental_fold_data_dict["X_comparison_train"])
    assert np.allclose(parental_fold_data_dict["X_dev"],parental_fold_data_dict["X_comparison_dev"])
    parental_train_set=MouseGeneAPADataset(parental_fold_data_dict["X_train"],
                                parental_fold_data_dict["X_indices_original_train"],
                                parental_fold_data_dict["Y_usage_train"],
                                parental_fold_data_dict["pas_numbers_train"],
                                batch_size=params["batch_size"],
                                shuffle=True)
    parental_train_set_comparison=MouseGeneAPADataset(parental_fold_data_dict["X_train"],
                                            parental_fold_data_dict["X_indices_original_train"],
                                            parental_fold_data_dict["Y_usage_train"],
                                            parental_fold_data_dict["pas_numbers_train"],
                                            batch_size=params["batch_size"],
                                            shuffle=False)
    parental_dev_set=MouseGeneAPADataset(parental_fold_data_dict["X_dev"],
                                parental_fold_data_dict["X_indices_original_dev"],
                                parental_fold_data_dict["Y_usage_dev"],
                                parental_fold_data_dict["pas_numbers_dev"],
                                batch_size=params["batch_size"],
                                shuffle=False)
    parental_dev_set_comparison=MouseGeneAPADataset(parental_fold_data_dict["X_dev"],
                                            parental_fold_data_dict["X_indices_original_dev"],
                                            parental_fold_data_dict["Y_usage_dev"],
                                            parental_fold_data_dict["pas_numbers_dev"],
                                            batch_size=params["batch_size"],
                                            shuffle=False)

    # create model
    model=APAModel(params)
    model.to(params["device"])
    optimizer=optim.Adam(model.parameters(),
                        lr=params["lr"])
    
    # define training loop
    local_step=0
    running_loss=0
    def parental_train(epoch):
        global local_step
        global running_loss
        model.train()
        parental_train_set.set_shuffle(True)
        for local_batch, local_labels,local_pas_numbers in parental_train_set:
            local_batch=(local_batch).to(params["device"])
            local_labels=(local_labels).to(params["device"])
            local_pas_numbers=local_pas_numbers.to(params["device"])
            optimizer.zero_grad()
            local_outputs=model(local_batch,local_pas_numbers)
            local_loss=loss_function(local_outputs,local_labels.type(torch.float32),local_pas_numbers,model,params)
            local_loss.backward()
            optimizer.step()
            local_step+=1
            running_loss += local_loss.item()
            if local_step % 20 == 0:    
                print('[%d, %5d] loss: %.3f' %
                    (epoch, local_step + 1, running_loss / 20))
                running_loss = 0.0
    
    # parental evaluation functions
    def parental_train_set_eval(epoch):
        name="Parental Train"
        phase="train"
        cross_entropy_eval(model,epoch,name,parental_train_set,params)
        mae_eval(model,epoch,name,parental_train_set,params)
        comparison_eval(model,epoch,name,parental_train_set_comparison,parental_fold_data_dict,params,phase)
        max_pred_eval(model,epoch,name,parental_train_set,params)

    def parental_dev_set_eval(epoch):
        name="Parental Dev"
        phase="dev"
        global best_dev_mae_value
        cross_entropy_eval(model,epoch,name,parental_dev_set,params)
        dev_mae_value=mae_eval(model,epoch,name,parental_dev_set,params)
        if epoch>0 and dev_mae_value<best_dev_mae_value:
            best_dev_mae_value=dev_mae_value
            state_dict=model.state_dict()
            torch.save(state_dict,params["parental_model_file"])
            print("saving model...")
        comparison_eval(model,epoch,name,parental_dev_set_comparison,parental_fold_data_dict,params,phase)
        max_pred_eval(model,epoch,name,parental_dev_set,params)   
    
    # begin parental training
    parental_train_set_eval(0)
    parental_dev_set_eval(0) 

    for epoch in range(1,10):
        if epoch==1:
            best_dev_mae_value=float("+inf")
        # Training
        parental_train(epoch)
        # Validation
        parental_train_set_eval(epoch)
        parental_dev_set_eval(epoch)
    
    print("reload the best model and test")
    model.load_state_dict(torch.load(params["parental_model_file"]))
    parental_dev_set_eval(-1)

    # F1 dataset loaders
    f1_fold_data_dict={}
    fold_train=fold_combine([f1_combined_data_dict_folds[(params["fold"]+i)%5] for i in [0,1,2,3]],'train')
    fold_test=fold_combine([f1_combined_data_dict_folds[(params["fold"]+i)%5] for i in [4]],'dev')
    f1_fold_data_dict.update(fold_train)
    f1_fold_data_dict.update(fold_test)
    # TODO: make dataset that contains pas_numbers_comparison and remove these two assertions
    assert np.allclose(f1_fold_data_dict["X_train"],f1_fold_data_dict["X_comparison_train"])
    assert np.allclose(f1_fold_data_dict["X_dev"],f1_fold_data_dict["X_comparison_dev"])
    f1_train_set=MouseGeneAPADataset(f1_fold_data_dict["X_train"],
                                f1_fold_data_dict["X_indices_original_train"],
                                f1_fold_data_dict["Y_usage_train"],
                                f1_fold_data_dict["pas_numbers_train"],
                                batch_size=params["batch_size"],
                                shuffle=True)
    f1_train_set_comparison=MouseGeneAPADataset(f1_fold_data_dict["X_train"],
                                            f1_fold_data_dict["X_indices_original_train"],
                                            f1_fold_data_dict["Y_usage_train"],
                                            f1_fold_data_dict["pas_numbers_train"],
                                            batch_size=params["batch_size"],
                                            shuffle=False)
    f1_dev_set=MouseGeneAPADataset(f1_fold_data_dict["X_dev"],
                                f1_fold_data_dict["X_indices_original_dev"],
                                f1_fold_data_dict["Y_usage_dev"],
                                f1_fold_data_dict["pas_numbers_dev"],
                                batch_size=params["batch_size"],
                                shuffle=False)
    f1_dev_set_comparison=MouseGeneAPADataset(f1_fold_data_dict["X_dev"],
                                            f1_fold_data_dict["X_indices_original_dev"],
                                            f1_fold_data_dict["Y_usage_dev"],
                                            f1_fold_data_dict["pas_numbers_dev"],
                                            batch_size=params["batch_size"],
                                            shuffle=False)
    # redifine optimizer to fine-tune on F1
    f1_optimizer=optim.Adam([v for k,v in model.named_parameters() if not k.startswith("conv1d")],
                    lr=1e-4)
    
    # F1 training loop
    local_step=0
    running_loss=0
    def f1_train(epoch):
        global local_step
        global running_loss
        model.train()
        parental_train_set.set_shuffle(True)
        for local_batch, local_labels,local_pas_numbers in f1_train_set:
            local_batch=(local_batch).to(params["device"])
            local_labels=(local_labels).to(params["device"])
            local_pas_numbers=local_pas_numbers.to(params["device"])
            f1_optimizer.zero_grad()
            local_outputs=model(local_batch,local_pas_numbers)
            local_loss=loss_function(local_outputs,local_labels.type(torch.float32),local_pas_numbers,model,params)
            local_loss.backward()
            f1_optimizer.step()
            local_step+=1
            running_loss += local_loss.item()
            if local_step % 20 == 0:    
                print('[%d, %5d] loss: %.3f' %
                    (epoch, local_step + 1, running_loss / 20))
                running_loss = 0.0 

    # F1 evaluation functions 
    def f1_train_set_eval(epoch):
        name="F1 Train"
        phase="train"
        cross_entropy_eval(model,epoch,name,f1_train_set,params)
        mae_eval(model,epoch,name,f1_train_set,params)
        comparison_eval(model,epoch,name,f1_train_set_comparison,f1_fold_data_dict,params,phase)
        max_pred_eval(model,epoch,name,f1_train_set,params)
    def f1_dev_set_eval(epoch):
        name="F1 Dev"
        phase="dev"
        global best_dev_mae_value
        cross_entropy_eval(model,epoch,name,f1_dev_set,params)
        dev_mae_value=mae_eval(model,epoch,name,f1_dev_set,params)
        if epoch>1 and dev_mae_value<best_dev_mae_value :
            best_dev_mae_value=dev_mae_value
            state_dict=model.state_dict()
            torch.save(state_dict,params["f1_model_file"])
            print("saving model...")
        comparison_eval(model,epoch,name,f1_dev_set_comparison,f1_fold_data_dict,params,phase)
        max_pred_eval(model,epoch,name,f1_dev_set,params) 

    # begin training of F1
    f1_train_set_eval(0)
    f1_dev_set_eval(0) 
    for epoch in range(1,10):
        if epoch==1:
            best_dev_mae_value=float("+inf")
        # Training
        f1_train(epoch)
        # Validation
        f1_train_set_eval(epoch)
        f1_dev_set_eval(epoch)
    
    print("reload the best F1 model and test")
    model.load_state_dict(torch.load(params["f1_model_file"]))
    f1_dev_set_eval(-1)


