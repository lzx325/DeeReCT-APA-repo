import numpy as np
import re
import itertools
def iupac2regex(s):
    """change the IUPAC symbols, e.g. 'U' 'R' 'Y', to equivalent regex
    """

    s=re.subn('U','T',s)[0]
    s=re.subn('R','[AG]',s)[0]
    s=re.subn('Y','[CT]',s)[0]
    s=re.subn('S','[GC]',s)[0]
    s=re.subn('W','[AT]',s)[0]
    s=re.subn('K','[GT]',s)[0]
    s=re.subn('M','[AC]',s)[0]
    s=re.subn('B','[CGT]',s)[0]
    s=re.subn('D','[AGT]',s)[0]
    s=re.subn('H','[ACT]',s)[0]
    s=re.subn('V','[ACG]',s)[0]
    s=re.subn('N','[ACGT]',s)[0]
    return s

# define following known motifs

polya_signals=[iupac2regex(s) for s in ["AATAAA", "ATTAAA", "TATAAA", "AGTAAA", "AAGAAA", "AATATA", "AATACA", "CATAAA", "GATAAA", "AATGAA", "TTTAAA", "ACTAAA", "AATAGA", "AAAAAG", "AAAATA", "GGGGCT", "AAAAAA", "ATAAAA", "AAATAA", "ATAAAT", "TTTTTT", "ATAAAG", "TAAAAA", "CAATAA", "TAATAA", "ATAAAC"]]
aue_elements=[iupac2regex(s) for s in ["GGGGAG" , "GUGGGG" , "GGGUGG", "UUUGUA" , "GUAUUU" , "CUGUGU", "UAUAUA" , "AUAUAU" , "UUUAUA", "UGUAUA" , "AUGUAU" , "UGUAUU" ]]
cue_elements=[iupac2regex(s) for s in ["UAUUUU" , "UGUUUU" , "UUUUUU","AAUAAA" , "AUAAAG" , "AAAUAA" ]]
cde_elements=[iupac2regex(s) for s in ["GUGUCU", "CUGCCU", "UGUCUC", "UUAUUU" , "UUUCUU" , "UGUUUU", "UGUGUG" , "GUGUGU", "CUGUGU", "CUGGGG" , "UGUCUG" , "GUCUGU"]]
ade_elements=[iupac2regex(s) for s in ["CCUCCC" , "CUCCCC", "CACCCC", "CCCGCC", "CCCCGC", "CCCGCG", "GGUGGG", "GGCUGG", "GGGUGG", "GGGCAG", "GGCCAG", "GGGGCC", "GGGAGG", "GGAGGG", "GGGGAG"]]
rbp_motifs=[iupac2regex(s) for s in [ "UUUUAU",  "GGGAGG",  "GGAGGG",  "GCUUGC",  "YGCY", "YGCUKY",  "ARAAGA",  "UUUUCU",  "UCAY",  "CCWWHCC",  "CCYYCCH", "UGGGRAD",  "GGGA",  "UKKGGK",  "GGSKG",  "UGUA", "UGUGU",  "GAAGAA"]]
all_1mer=[''.join(t) for t in itertools.product('ATGC',repeat=1)]
all_2mer=[''.join(t) for t in itertools.product('ATGC',repeat=2)]
all_3mer=[''.join(t) for t in itertools.product('ATGC',repeat=3)]
all_4mer=[''.join(t) for t in itertools.product('ATGC',repeat=4)]

def get_region(s):
    mid=len(s)//2
    return mid-40,mid,mid+40

def get_count_feature(s,motifs,lo,hi):
    if lo>=hi:
        return [0 for _ in motifs]
    lo=max(lo,0)
    hi=min(hi,len(s))
    features=[]
    for p in motifs:
        features.append(s[lo:hi].count(p))
    return features
def get_count_feature_regex(s,motifs,lo,hi):
    if lo>=hi:
        return [0 for _ in motifs]
    lo=max(lo,0)
    hi=min(hi,len(s))
    features=[]
    for p in motifs:
        features.append(re.subn(p,'',s[lo:hi])[1])
    return features
def polya_signal_features(s):
    loc1,loc2,loc3=get_region(s)
    features1=get_count_feature(s,polya_signals,0,loc1)
    features2=get_count_feature(s,polya_signals,loc1,loc2)
    return features1+features2
def aue_features(s):
    loc1,loc2,loc3=get_region(s)
    features=get_count_feature(s,aue_elements,0,loc1)
    return features
def cue_features(s):
    loc1,loc2,loc3=get_region(s)
    features=get_count_feature(s,cue_elements,loc1,loc2)
    return features
def cde_features(s):
    loc1,loc2,loc3=get_region(s)
    features=get_count_feature(s,cde_elements,loc2,loc3)
    return features
def ade_features(s):
    loc1,loc2,loc3=get_region(s)
    features=get_count_feature(s,ade_elements,loc3,len(s))
    return features
def rbp_features(s):
    loc1,loc2,loc3=get_region(s)
    features1=get_count_feature_regex(s,rbp_motifs,0,loc1)
    features2=get_count_feature_regex(s,rbp_motifs,loc1,loc2)
    features3=get_count_feature_regex(s,rbp_motifs,loc2,loc3)
    features4=get_count_feature_regex(s,rbp_motifs,loc3,len(s))
    return features1+features2+features3+features4

def mer_features(s,mer):
    motifs=None
    if mer=='one':
        motifs=all_1mer
    elif mer=='two':
        motifs=all_2mer
    elif mer=='three':
        motifs=all_3mer
    elif mer=='four':
        motifs=all_4mer
    else:
        assert False
    loc1,loc2,loc3=get_region(s)
    features1=get_count_feature(s,motifs,0,loc1)
    features2=get_count_feature(s,motifs,loc1,loc2)
    features3=get_count_feature(s,motifs,loc2,loc3)
    features4=get_count_feature(s,motifs,loc3,len(s))
    return features1+features2+features3+features4
def get_all_features(s):
    """
    s: string
        one nucleic acid sequence for feature extraction

    Returns

    features: list
        extracted features

    """
    features=[]
    features.extend(polya_signal_features(s))
    features.extend(aue_features(s))
    features.extend(cue_features(s))
    features.extend(cde_features(s))
    features.extend(ade_features(s))
    features.extend(rbp_features(s))
    features.extend(mer_features(s,'one'))
    features.extend(mer_features(s,'two'))
    features.extend(mer_features(s,'three'))
    features.extend(mer_features(s,'four'))
    return features
