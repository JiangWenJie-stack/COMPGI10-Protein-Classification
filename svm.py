# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import tensorflow as tf
import functools


from Bio import SeqIO

# read the fasta files
mito = list(SeqIO.parse("mito.fasta", "fasta"))
secreted = list(SeqIO.parse("secreted.fasta", "fasta"))
nucleus = list(SeqIO.parse("nucleus.fasta", "fasta"))
cyto = list(SeqIO.parse("cyto.fasta", "fasta"))

# fill the sequences with just character arrays 
mitoSeq=[]
secSeq=[]
nucSeq=[]
cytoSeq=[]

for seq in mito:
    mitoSeq.append(str(seq.seq))
for seq in secreted:
    secSeq.append(str(seq.seq))
for seq in nucleus:
    nucSeq.append(str(seq.seq))
for seq in cyto:
    cytoSeq.append(str(seq.seq))    

print(mitoSeq[0])

mitoLab=np.zeros((len(mitoSeq)))
secLab=np.zeros((len(secSeq)))
nucLab=np.zeros((len(nucSeq)))
cytoLab=np.zeros((len(cytoSeq)))
mitoLab[:]=1
secLab[:]=2
nucLab[:]=3
cytoLab[:]=4



# set random seeds
np.random.seed(80085)
tf.set_random_seed(80085)

# Switches for training and optimizing
TRAIN_MODE = False
GRID_SEARCH = False




### Function as per https://danijar.com/structuring-your-tensorflow-models/











