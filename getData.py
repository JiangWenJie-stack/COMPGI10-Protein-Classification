# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:03:03 2017

@author: lewismoffat
"""
# imports    
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict

import pdb


# function for retrieving data from fasta formated in different ways
def getData(t="train", setup=1, limit=333):
    # potential files types
    files=["mito.fasta","cyto.fasta","nucleus.fasta","secreted.fasta"]
    # get the training data
    if t=="train":
        path="train/"
    else:# get the test data
        path="test/"
        
        
    # quick if statesments to decide on what mode its in
    if setup==1:
        # this gets the ProFET implementation which makes a call to the library
        print("not ready")
    elif setup==2:
        # this is the p-tuple set up, defaulted to 3
        sq=[]
        labels=[]
        for idx,file in enumerate(files):
            seqs = pTupleData(path+file)
            label = np.zeros((len(seqs),4))
            label[:,idx]=1
            sq.append(seqs)
            labels.append(label)
        
        sq=sq[0]+sq[1]+sq[2]+sq[3]
        # encodes feature dictionaries as numpy vectors, needed by scikit-learn.
        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform([event_feat(x) for x in sq])        
        return X, np.concatenate(labels[:])
        
        
    elif setup==3:
        # this is the normal set up for start and end values one hot
        # loop through files
        sq=[]
        sql=[]
        labels=[]
        for idx,file in enumerate(files):
            seqs, seqsL = oneHotEasy(path+file, limit)
            label = np.zeros((len(seqs),4))
            label[:,idx]=1
            sq.append(seqs)
            sql.append(seqsL)
            labels.append(label)
        return np.concatenate(sq[:]), np.concatenate(sql[:]), np.concatenate(labels[:])
    elif setup==4:
        # this replaces characters with integer ID's
        # loop through files
        sq=[]
        sql=[]
        labels=[]
        for idx,file in enumerate(files):
            seqs, seqsL = intID(path+file, limit)
            label = np.zeros((len(seqs),4))
            label[:,idx]=1
            sq.append(seqs)
            sql.append(seqsL)
            labels.append(label)
        return np.concatenate(sq[:]), np.concatenate(sql[:]), np.concatenate(labels[:])
    elif setup==5:
        # this replaces characters with integer ID's includes the tails
        # loop through files
        sq=[]
        sql=[]
        labels=[]
        for idx,file in enumerate(files):
            seqs, seqsL = intID2(path+file, limit)
            label = np.zeros((len(seqs),4))
            label[:,idx]=1            
            sq.append(seqs)
            sql.append(seqsL)
            labels.append(label)
        return np.concatenate(sq[:]), np.concatenate(sql[:]), np.concatenate(labels[:])
    else:
        print("please provide a valid setup number e.g. 1,2,3,4")
    
    """ 
    setup is either 1,2,3,4
    1: get ProFET features 
    2: get ptuples
    3: get onehot of first and last 100 with padding in between if needed
    4: get onehot of up to first 333 and return sequence length as well
    
    """

    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# HELPERS

# function to take a fasta file and import it to 300 value sequences that are one hot
def oneHotEasy(fasta, limit):
    # putting the dictionary here so it doesnt just sit in memory latter
    intDict={'A':np.array([1]),
          'C':np.array([2]),
          'D':np.array([3]),
          'E':np.array([4]),
          'F':np.array([5]),
          'G':np.array([6]),
          'H':np.array([7]),
          'I':np.array([8]),
          'K':np.array([9]),
          'L':np.array([10]),
          'M':np.array([11]),
          'N':np.array([12]),
          'P':np.array([13]),
          'Q':np.array([14]),
          'R':np.array([15]),
          'S':np.array([16]),
          'T':np.array([17]),
          'V':np.array([18]),
          'W':np.array([19]),
          'Y':np.array([20]),
          'X':np.array([21]),
          'Z':np.array([21]),
          'U':np.array([21]),
          'B':np.array([21]),
          '0':np.array([21])}

    # fasta is the file path to fasta file eg mito.fasta
    listFast = list(SeqIO.parse(fasta, "fasta"))
    # use the sklearn onehot encoder to deal with things
    enc=OneHotEncoder()
    enc.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21]])
    seqs=[]
    seqL=[]
    for seq in listFast:
        stseq=str(seq.seq)
        
        if len(stseq)>limit:
            # clips to max 333 length
            stseq=stseq[:limit]
            seqL.append(limit)
        else:
            # pad with zeros to get to length 333
            seqL.append(len(stseq))
            stseq=stseq.ljust(limit,'0')
        numseq=[]
        for char in stseq:
            numseq.append(intDict[char])
            
        numseq=enc.transform(numseq)
        seqs.append(np.expand_dims(numseq.toarray(),0))
    return np.concatenate(seqs[:]), seqL

def pTuple(vec,n=3):
    n=3
    return [vec[i:i+n] for i in range(len(vec)-n+1)]

def event_feat(event):
    ####### Creates Dictionary ########
    result = defaultdict(float)
    for tup in event:
        if "X" in tup or "Z" in tup or "U" in tup :
            continue
        result[tup]+=1
    return result

def pTupleData(fasta):

    # fasta is the file path to fasta file eg mito.fasta
    listFast = list(SeqIO.parse(fasta, "fasta"))
    # use the sklearn onehot encoder to deal with things
    seqs=[]
    for seq in listFast:
        stseq=str(seq.seq)
        stseq = stseq.replace("B", "")
        s1=pTuple(stseq)
        seqs.append(s1)
        
    return seqs
                          
def intID(fasta, limit):
    # putting the dictionary here so it doesnt just sit in memory latter
    intDict={'A':np.array([1]),
              'C':np.array([2]),
              'D':np.array([3]),
              'E':np.array([4]),
              'F':np.array([5]),
              'G':np.array([6]),
              'H':np.array([7]),
              'I':np.array([8]),
              'K':np.array([9]),
              'L':np.array([10]),
              'M':np.array([11]),
              'N':np.array([12]),
              'P':np.array([13]),
              'Q':np.array([14]),
              'R':np.array([15]),
              'S':np.array([16]),
              'T':np.array([17]),
              'V':np.array([18]),
              'W':np.array([19]),
              'Y':np.array([20]),
              'X':np.array([21]),
              'Z':np.array([21]),
              'U':np.array([21]),
              'B':np.array([21]),
              '0':np.array([21])}

    # fasta is the file path to fasta file eg mito.fasta
    listFast = list(SeqIO.parse(fasta, "fasta"))
    seqs=[]
    seqL=[]
    for seq in listFast:
        stseq=str(seq.seq)
        if len(stseq)>limit:
            # clips to max 333 length
            stseq=stseq[:limit]
            seqL.append(limit)
        else:
            # pad with zeros to get to length 333
            seqL.append(len(stseq))
            stseq=stseq.ljust(limit,'0')
        numseq=[]
        for char in stseq:
            numseq.append(intDict[char])
        seqs.append(numseq)
    return np.squeeze(np.array(seqs)), seqL
                          
def intID2(fasta, limit):
    # putting the dictionary here so it doesnt just sit in memory latter
    intDict={'A':1,
          'C':2,
          'D':3,
          'E':4,
          'F':5,
          'G':6,
          'H':7,
          'I':8,
          'K':9,
          'L':10,
          'M':11,
          'N':12,
          'P':13,
          'Q':14,
          'R':15,
          'S':16,
          'T':17,
          'V':18,
          'W':19,
          'Y':20,
          'X':21,
          'Z':21,
          'U':21,
          'B':21,
          '0':21}
    tail=30
    # fasta is the file path to fasta file eg mito.fasta
    listFast = list(SeqIO.parse(fasta, "fasta"))
    seqs=[]
    seqL=[]
    for seq in listFast:
        stseq=str(seq.seq)
        if len(stseq)>(limit):
            # clips to max 333 length
            stseq=stseq[:limit]+stseq[-tail:]
            seqL.append(limit+tail)
        else:
            # pad with zeros to get to length 333
            seqL.append(len(stseq)+tail)
            newt=stseq[-tail:]
            if len(newt)!=tail:
                newt=newt.ljust(tail,'0')
            stseq=stseq.ljust(limit,'0')+newt
        numseq=[]
        for char in stseq:
            numseq.append(intDict[char])
        seqs.append(numseq)    
    return np.array(seqs), seqL
  
# testing
#sq,labels=getData(t="train", setup=2)