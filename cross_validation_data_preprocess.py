#!/usr/bin/env python



import os
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
from pas_utils import *
np.random.seed(3)

if __name__=="__main__":
    OUTPUT_DIR="./APA_ML/processed"
    
    PARENTAL_SEQUENCE_TABLE_FILE=os.path.join(OUTPUT_DIR,'parental_sequence_table.h5')
    PARENTAL_BL_PROCESSED_SEQUENCE_FILE=os.path.join(OUTPUT_DIR,'processed_parental_bl_sequences.npy')
    PARENTAL_SP_PROCESSED_SEQUENCE_FILE=os.path.join(OUTPUT_DIR,'processed_parental_sp_sequences.npy')
    PARENTAL_BL_PROCESSED_FEATURES_FILE=os.path.join(OUTPUT_DIR,'processed_parental_bl_features.npy')
    PARENTAL_SP_PROCESSED_FEATURES_FILE=os.path.join(OUTPUT_DIR,'processed_parental_sp_features.npy')


    F1_SEQUENCE_TABLE_FILE=os.path.join(OUTPUT_DIR,'f1_sequence_table.h5')
    F1_BL_PROCESSED_SEQUENCE_FILE=os.path.join(OUTPUT_DIR,'processed_f1_bl_sequences.npy')
    F1_SP_PROCESSED_SEQUENCE_FILE=os.path.join(OUTPUT_DIR,'processed_f1_sp_sequences.npy')
    F1_BL_PROCESSED_FEATURES_FILE=os.path.join(OUTPUT_DIR,'processed_f1_bl_features.npy')
    F1_SP_PROCESSED_FEATURES_FILE=os.path.join(OUTPUT_DIR,'processed_f1_sp_features.npy')


    parental_sequence_table=pd.read_hdf(PARENTAL_SEQUENCE_TABLE_FILE)
    parental_bl_processed_sequences=np.load(PARENTAL_BL_PROCESSED_SEQUENCE_FILE)
    parental_sp_processed_sequences=np.load(PARENTAL_SP_PROCESSED_SEQUENCE_FILE)
    parental_gene_ids=list(np.load(os.path.join(OUTPUT_DIR,'parental_gene_ids_shuffled.npy')))
    parental_bl_processed_features=np.load(PARENTAL_BL_PROCESSED_FEATURES_FILE)
    parental_sp_processed_features=np.load(PARENTAL_SP_PROCESSED_FEATURES_FILE)



    f1_sequence_table=pd.read_hdf(F1_SEQUENCE_TABLE_FILE)
    f1_bl_processed_sequences=np.load(F1_BL_PROCESSED_SEQUENCE_FILE)
    f1_sp_processed_sequences=np.load(F1_SP_PROCESSED_SEQUENCE_FILE)
    f1_gene_ids=list(np.load(os.path.join(OUTPUT_DIR,'f1_gene_ids_shuffled.npy')))
    f1_bl_processed_features=np.load(F1_BL_PROCESSED_FEATURES_FILE)
    f1_sp_processed_features=np.load(F1_SP_PROCESSED_FEATURES_FILE)


    data_source=dict(parental_sequence_table=parental_sequence_table,
                    parental_bl_processed_sequences=parental_bl_processed_sequences,
                    parental_sp_processed_sequences=parental_sp_processed_sequences,
                    parental_gene_ids=parental_gene_ids,
                    parental_bl_processed_features=parental_bl_processed_features,
                    parental_sp_processed_features=parental_sp_processed_features,
                    f1_sequence_table=f1_sequence_table,
                    f1_bl_processed_sequences=f1_bl_processed_sequences,
                    f1_sp_processed_sequences=f1_sp_processed_sequences,
                    f1_gene_ids=f1_gene_ids,
                    f1_bl_processed_features=f1_bl_processed_features,
                    f1_sp_processed_features=f1_sp_processed_features)



    print("prepare sequence folds")
    for generation in ['parental','f1']:
        for strain in ['bl','sp']:
            for attributes in ['sequences']:
                for final_length in [455]:
                    print('\n%s %s %s %s'%(generation,strain,attributes,final_length))
                    if generation=='parental':
                        fold_size=1651
                    elif generation=='f1':
                        fold_size=439
                    prepare_fold_files(data_source,generation,strain,attributes,5,fold_size,[],OUTPUT_DIR,final_length)
    print()



    print("prepare feature folds")
    for generation in ['parental','f1']:
        for strain in ['bl','sp']:
            for attributes in ['features']:
                print('\n%s %s %s'%(generation,strain,attributes))
                if generation=='parental':
                    fold_size=1651
                elif generation=='f1':
                    fold_size=439
                prepare_fold_files(data_source,generation,strain,attributes,5,fold_size,[],OUTPUT_DIR)
    print()


