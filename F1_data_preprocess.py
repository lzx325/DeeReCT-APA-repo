#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from itertools import chain
import pickle
from pas_utils import *
from feature import *
if __name__=="__main__":
    OUTPUT_DIR="./APA_ML/processed"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    CONTROL_USAGE_FILE='./APA_ML/F1/control.f1.usage.txt'
    DIFFERENTIAL_USAGE_FILE='./APA_ML/F1/differential.f1.usage.txt'
    PARENTAL_SEQUENCE_TABLE_FILE=os.path.join(OUTPUT_DIR,'parental_sequence_table.h5')


    control_usage_table=pd.read_table(CONTROL_USAGE_FILE)
    differential_usage_table=pd.read_table(DIFFERENTIAL_USAGE_FILE)
    parental_table=pd.read_hdf(PARENTAL_SEQUENCE_TABLE_FILE)

    control_usage_table.rename(map_index,inplace=True)
    differential_usage_table.rename(map_index,inplace=True)

    control_sequence_table=pd.DataFrame({'coordinate':parental_table['coordinate'].loc[control_usage_table.index],
                                        'bl_sequence':parental_table['bl_sequence'].loc[control_usage_table.index],
                                        'sp_sequence':parental_table['sp_sequence'].loc[control_usage_table.index],
                                        'bl_usage':control_usage_table['F1/BL6'],
                                        'sp_usage':control_usage_table['F1/SPR'],
                                        'differential':False})
    control_sequence_table=control_sequence_table[['coordinate','bl_sequence','sp_sequence','bl_usage','sp_usage','differential']]
    differential_sequence_table=pd.DataFrame({'coordinate':parental_table['coordinate'].loc[differential_usage_table.index],
                                            'bl_sequence':parental_table['bl_sequence'].loc[differential_usage_table.index],
                                            'sp_sequence':parental_table['sp_sequence'].loc[differential_usage_table.index],
                                            'bl_usage':differential_usage_table['F1/BL6'],
                                            'sp_usage':differential_usage_table['F1/SPR'],
                                            'differential':True})
    differential_sequence_table=differential_sequence_table[['coordinate','bl_sequence','sp_sequence','bl_usage','sp_usage','differential']]
    sequence_table=pd.concat([control_sequence_table,differential_sequence_table])


    gene_ids=list(sorted(set([pas_id.split(':')[0] for pas_id in sequence_table.index])))

    sequence_table.sort_index(inplace=True)

    print("Preparing Signals")
    bl_signal=[]
    sp_signal=[]
    for i,gene in enumerate(gene_ids):
        gene_indices=sequence_table.index.str.startswith(gene)
        print("[%d/%d]"%(i+1,len(gene_ids)),end='\r')
        gene_bl_usage=sequence_table.loc[gene_indices].sort_index()['bl_usage'].values
        gene_sp_usage=sequence_table.loc[gene_indices].sort_index()['sp_usage'].values
        gene_bl_signal=usage2signal(gene_bl_usage)
        gene_sp_signal=usage2signal(gene_sp_usage)
        bl_signal.append(gene_bl_signal)
        sp_signal.append(gene_sp_signal)
    bl_signal=np.concatenate(bl_signal,axis=0)
    sp_signal=np.concatenate(sp_signal,axis=0)
    print()
    sorted_index=sequence_table.sort_index().index
    bl_signal=pd.Series(bl_signal,index=sorted_index)
    sp_signal=pd.Series(sp_signal,index=sorted_index)
    sequence_table['bl_signal']=bl_signal
    sequence_table['sp_signal']=sp_signal

    sequence_table.loc[sequence_table.bl_signal>1,"bl_signal"]=1
    sequence_table.loc[sequence_table.sp_signal>1,"sp_signal"]=1

    sequence_table.to_hdf(os.path.join(OUTPUT_DIR,'f1_sequence_table.h5'),key='key',mode='w')

    gene_ids=np.array(gene_ids)
    np.random.shuffle(gene_ids)
    np.save(os.path.join(OUTPUT_DIR,'f1_gene_ids_shuffled.npy'),gene_ids)

    MAX_SEQ_LEN=max(chain(map(lambda s:len(s),sequence_table['bl_sequence']),map(lambda s:len(s),sequence_table['sp_sequence'])))



    pas_ids=list(sequence_table.index)
    bl_sequences=list(sequence_table['bl_sequence'])
    sp_sequences=list(sequence_table['sp_sequence'])
    processed_bl_sequences=sequence_process(bl_sequences,MAX_SEQ_LEN)
    processed_sp_sequences=sequence_process(sp_sequences,MAX_SEQ_LEN)

    print("Preparing parental BL features")
    processed_bl_features=feature_process(bl_sequences,MAX_SEQ_LEN)
    print()
    print("preparing parental SP features")
    processed_sp_features=feature_process(sp_sequences,MAX_SEQ_LEN)
    print()
    processed_bl_features_norm=(processed_bl_features-processed_bl_features.mean(axis=0))/processed_bl_features.std(axis=0)
    processed_sp_features_norm=(processed_sp_features-processed_sp_features.mean(axis=0))/processed_sp_features.std(axis=0)

    np.save(os.path.join(OUTPUT_DIR,'processed_f1_bl_sequences.npy'),processed_bl_sequences,allow_pickle=False)
    np.save(os.path.join(OUTPUT_DIR,'processed_f1_sp_sequences.npy'),processed_sp_sequences,allow_pickle=False)


    np.save(os.path.join(OUTPUT_DIR,'processed_f1_bl_features.npy'),processed_bl_features_norm,allow_pickle=False)
    np.save(os.path.join(OUTPUT_DIR,'processed_f1_sp_features.npy'),processed_sp_features_norm,allow_pickle=False)




