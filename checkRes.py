from Bio import SeqIO
import csv
import pdb
import numpy as np

testingDir='test' 

# read the fasta files
mito = list(SeqIO.parse("test/mito.fasta", "fasta"))
secreted = list(SeqIO.parse("test/secreted.fasta", "fasta"))
nucleus = list(SeqIO.parse("test/nucleus.fasta", "fasta"))
cyto = list(SeqIO.parse("test/cyto.fasta", "fasta"))

with open(testingDir+'/PredictedTestSetResults.csv', 'r') as f:
    reader = csv.reader(f)
    Testing = list(reader)


mitoL = []
secretedL = [] 
nucleusL = []
cytoL = []

for seq in mito:
    mitoL.append([seq.id,'mito'])
for seq in secreted:
    secretedL.append([seq.id,'secreted'])
for seq in nucleus:
    nucleusL.append([seq.id,'nucleus'])
for seq in cyto:
    cytoL.append([seq.id,'cyto'])
actual = mitoL+secretedL+nucleusL+cytoL

rez=[]

for i, j in enumerate(actual):
    for a,b in enumerate(Testing):
        if j[0]==b[0]:
            rez.append([j[1],b[1]])
rez=np.array(rez)       
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print(classification_report(rez[:,0], rez[:,1]))
print()
print('Accuracy score:')
print(accuracy_score(rez[:,0], rez[:,1]))