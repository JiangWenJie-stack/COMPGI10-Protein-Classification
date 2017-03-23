# -*- coding: utf-8 -*-

import numpy as np
from Bio import SeqIO

# read the fasta files
mito = list(SeqIO.parse("data/mito.fasta", "fasta"))
secreted = list(SeqIO.parse("data/secreted.fasta", "fasta"))
nucleus = list(SeqIO.parse("data/nucleus.fasta", "fasta"))
cyto = list(SeqIO.parse("data/cyto.fasta", "fasta"))

np.random.shuffle(mito)
np.random.shuffle(secreted)
np.random.shuffle(nucleus)
np.random.shuffle(cyto)

"""
Number of Examples in each file
mito     = 1299  [325]
secreted = 1605  [401]
nucleus  = 3314  [829]
cyto     = 3004  [751]

Make test set 25%
Shuffle values then split then

"""
# mito test train
sequences = mito[:325]  # add code here
with open("test/mito.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
sequences = mito[325:]  # add code here
with open("train/mito.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
    
# secreted test train
sequences = secreted[:401]  # add code here
with open("test/secreted.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
sequences = secreted[401:]  # add code here
with open("train/secreted.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
    
# nucleus test train
sequences = nucleus[:829]  # add code here
with open("test/nucleus.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
sequences = nucleus[829:]  # add code here
with open("train/nucleus.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
    
# cyto test train
sequences = cyto[:751]  # add code here
with open("test/cyto.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
sequences = cyto[751:]  # add code here
with open("train/cyto.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")
    



