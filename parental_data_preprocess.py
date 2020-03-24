#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from itertools import chain
import pickle
from pas_utils import *
from feature import *

def table2dict(table):
    ret_dict=defaultdict(list)
    for index,row in table.iterrows():
        gene_id=index.split(':')[0]
        ret_dict[gene_id].append(dict(row))
    return ret_dict

def get_comparison_pairs(table,gene_ids):
    pair_list=[]
    for i,gene_id in enumerate(gene_ids):
        print(i,end='\r')
        gene_indices=table.index[table.index.str.startswith(gene_id)]
        for pas1,pas2 in combinations(gene_indices,2):
            if np.abs(table['bl_usage'][pas1]-table['sp_usage'][pas2])>0.05:
                pair_list.append((pas1,pas2))
if __name__=="__main__":
    OUTPUT_DIR="./APA_ML/processed"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    BL_SEQUENCE_FILE='./APA_ML/Parental/bl.pAs.sequence.txt'
    SP_SEQUENCE_FILE='./APA_ML/Parental/sp.pAs.sequence.txt'
    USAGE_FILE='./APA_ML/Parental/parental.pAs.usage.txt'
    SEQ_LEN=440

    sequence_table=pd.read_table(BL_SEQUENCE_FILE)
    sequence_table.rename(map_index,inplace=True)
    sequence_table.sort_index(inplace=True)


    sp_sequence_table=pd.read_table(SP_SEQUENCE_FILE)
    sp_sequence_table.rename(map_index,inplace=True)

    sequence_table.rename(columns={'sequence':'bl_sequence'},inplace=True)
    sequence_table['bl_sequence']=sequence_table['bl_sequence'].str.upper()
    sequence_table['sp_sequence']=sp_sequence_table['sequence'].str.upper()


    seq_dict=table2dict(sequence_table)
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
    gene_ids=list(sorted(set([pas_id.split(':')[0] for pas_id in sequence_table.index])))
    usage_table=pd.read_table(USAGE_FILE)
    usage_table.rename(map_index,inplace=True)
    sequence_table['bl_usage']=usage_table['BL']
    sequence_table['sp_usage']=usage_table['SP']
    sequence_table=sequence_table.fillna(value={'bl_usage':0,'sp_usage':0})

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
    bl_signal[np.isnan(bl_signal)]=0.5
    sp_signal[np.isnan(sp_signal)]=0.5
    print()
        
    sorted_index=sequence_table.sort_index().index
    bl_signal=pd.Series(bl_signal,index=sorted_index)
    sp_signal=pd.Series(sp_signal,index=sorted_index)
    sequence_table['bl_signal']=bl_signal
    sequence_table['sp_signal']=sp_signal

    sequence_table.loc[sequence_table.bl_signal>1,"bl_signal"]=1
    sequence_table.loc[sequence_table.sp_signal>1,"sp_signal"]=1

    sequence_table.to_hdf(os.path.join(OUTPUT_DIR,'parental_sequence_table.h5'),key='key',mode='w')

    np.save(os.path.join(OUTPUT_DIR,'processed_parental_bl_sequences.npy'),processed_bl_sequences,allow_pickle=False)
    np.save(os.path.join(OUTPUT_DIR,'processed_parental_sp_sequences.npy'),processed_sp_sequences,allow_pickle=False)

    np.save(os.path.join(OUTPUT_DIR,'processed_parental_bl_features.npy'),processed_bl_features_norm,allow_pickle=False)
    np.save(os.path.join(OUTPUT_DIR,'processed_parental_sp_features.npy'),processed_sp_features_norm,allow_pickle=False)

    gene_ids=np.array(gene_ids)
    np.random.shuffle(gene_ids)
    np.save(os.path.join(OUTPUT_DIR,'parental_gene_ids_shuffled.npy'),gene_ids)