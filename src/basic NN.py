import json
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import torch_geometric 


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


path_to_training = Path("dataset/all_training_data")
path_to_test = Path("dataset/test")

#####
# training and test sets of transcription ids
#####
training_set = ['ES2002']
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003']
test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

#####
# naive_baseline: all utterances are predicted important (label 1)
#####
test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    test_labels[transcription_id] = [1] * len(transcription)

with open("../submissions/test_labels_naive_baseline.json", "w") as file:
    json.dump(test_labels, file, indent=4)

#####
# text_baseline: utterances are embedded with SentenceTransformer, then train a classifier.
#####
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer

bert = SentenceTransformer('all-MiniLM-L6-v2')

y_training = []
with open("dataset/training_labels.json", "r") as file:
    training_labels = json.load(file)
X_training = []
for transcription_id in training_set:
    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    for utterance in transcription:
        X_training.append(utterance["speaker"] + ": " + utterance["text"])
    
    y_training += training_labels[transcription_id]

# The encoding gives 384 dimensions embeddings.
X_training = bert.encode(X_training, show_progress_bar=True)

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = torch.from_numpy(np.array(X_training)).float().to(device)
Y_train = torch.from_numpy(np.expand_dims(np.array(y_training), 1)).float().to(device)

n_features = 40
model = nn.Sequential(
    nn.Linear(384, n_features),
    nn.ReLU(),
    nn.Linear(n_features, 1),
    nn.Sigmoid()
)

model.to(device)

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #actually SGD is just GD in this case

losses = []
tr_acc = []

for epoch in range(5000):
    output = model.forward(X_train)
    loss = loss_function(output, Y_train)
    optimizer.zero_grad() #required since pytorch accumulates the gradients
    loss.backward() #backpropagation step
    optimizer.step() #update the parameters
    #update loss and accuracy
    losses.append(loss.data)
    tr_acc.append(accuracy_score(Y_train.cpu(), torch.max(output.cpu(), 1)[1]))

print("\n\nLa précision est de :")
print(tr_acc[-1])
print("\n")

test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    X_test = []
    for utterance in transcription:
        X_test.append(utterance["speaker"] + ": " + utterance["text"])
    
    X_test = bert.encode(X_test)
    X_test_tensor = torch.from_numpy(np.array(X_test)).float().to(device)

    y_test = np.round(model.forward(X_test_tensor).detach().numpy())
    test_labels[transcription_id] = [int(i) for i in flatten(y_test.tolist())]

with open("../submissions/test_labels.json", "w") as file:
    json.dump(test_labels, file, indent=4)
