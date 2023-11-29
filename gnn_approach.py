import numpy as np
import json
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from pathlib import Path
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from torch_geometric.data import Data
import networkx as nx

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

training_set = ['ES2002']
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('ES2002a')
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003']
test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

path_to_training = Path("training")
path_to_test = Path("test")

y_training = []

with open("training_labels.json", "r") as file:
    training_labels = json.load(file)
for train_id in training_set:
    y_training.append(training_labels[train_id])

def process_data(conversation_set, path):
    Bert_training = []
    E_training = [] # Contains edges of the graphs, with associated labels

    n_examples = len(conversation_set)

    print("||Extracting graph data||")
    for train_id in conversation_set:
        current_bert = []
        with open(path / f"{train_id}.json", "r") as file:
            transcription = json.load(file)

        for utterance in transcription:
            current_bert.append(utterance["speaker"] + ": " + utterance["text"])
        Bert_training.append(current_bert)

        with open(path / f"{train_id}.txt", "r") as file:
            lines = [line.rstrip() for line in file]
            edges = [line.split(" ") for line in lines]
            edges = [[edge[1], int(edge[0]), int(edge[2])] for edge in edges]
            E_training.append(edges)
    edge_tensor = [torch.LongTensor([[l[1], l[2]] for l in edge_set] + [[l[2], l[1]] for l in edge_set]).T for edge_set in E_training]
    edge_index = [np.array([[l[1], l[2]] for l in edge_set] + [[l[2], l[1]] for l in edge_set]) for edge_set in E_training]
    adjacency_lists = [[[l[1], l[2]] for l in edge_set] for edge_set in E_training]

    bert = SentenceTransformer('all-MiniLM-L6-v2')
    print("||Beginning encoding||")
    Embeddings_training = [bert.encode(el, show_progress_bar=False) for el in Bert_training]

    print("||Calculating features||")
    # Calculating useful node features
    graphs = [nx.Graph(l) for l in adjacency_lists]
    centralities = [nx.betweenness_centrality(G) for G in graphs]
    node_features = [torch.FloatTensor([[graphs[a].degree(i), centralities[a][i]] for i in range(graphs[a].number_of_nodes())]) for a in range(len(graphs))]

    # Calculating useful edge features
    edge_cosine_sim = [np.array([Embeddings_training[i][edge[0]] @ Embeddings_training[i][edge[1]] / (norm(Embeddings_training[i][edge[0]]) * norm(Embeddings_training[i][edge[1]])) for edge in edge_index[i]]) for i in range(n_examples)]
    edge_distance = [np.array([norm(Embeddings_training[i][edge[0]] - Embeddings_training[i][edge[1]]) for edge in edge_index[i]]) for i in range(n_examples)]
    edge_label = [np.array([l[0] for l in edge_set]) for edge_set in E_training]

    edge_attr = [torch.FloatTensor(np.column_stack([edge_cosine_sim[i], edge_distance[i]])) for i in range(n_examples)]

    data = [Data(x=node_features[i], edge_index=edge_tensor[i], edge_attr=edge_attr[i]) for i in range(n_examples)]

    return data

data_training = process_data(training_set, path_to_training)
n_features = data_training[0].x.shape[1]

model = gnn.Sequential('x, edge_index, edge_attr,', [
    (gnn.ResGatedGraphConv(2, 10, edge_dim = 2, act = nn.Sigmoid()), 'x, edge_index, edge_attr -> x'),
    (gnn.ResGatedGraphConv(10, 10, edge_dim = 2, act = nn.Sigmoid()), 'x, edge_index, edge_attr -> x'),
    gnn.Linear(10, 1),
    nn.Sigmoid()
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #actually SGD is just GD in this case

losses = []
tr_acc = []

print("||Training model||")
for epoch in range(1000):
    for i in range(len(data_training)):
        dataset = data_training[i].to(device)
        Y_train = torch.FloatTensor(y_training[i]).to(device).unsqueeze(1)
        output = model(dataset.x, dataset.edge_index, dataset.edge_attr)

        loss_function = nn.BCELoss(weight=(9*Y_train+1))
        loss = loss_function(output, Y_train)
        optimizer.zero_grad() #required since pytorch accumulates the gradients
        loss.backward() #backpropagation step
        optimizer.step() #update the parameters

        #update loss and accuracy
        losses.append(loss.data)
        tr_acc.append(accuracy_score(Y_train.cpu(), torch.max(output.cpu(), 1)[1]))

        dataset.detach()
        Y_train.detach()

print("\n\nLa précision est de :")
print(tr_acc[-1])
print("\n")

print("||Evaluating model||")
data_test = process_data(test_set, path_to_test)

n_tests = len(data_test)

test_labels = {}
for i in range(n_tests):
    dataset = data_test[i].to(device)

    y_test_expected = np.round(model(dataset.x, dataset.edge_index, dataset.edge_attr).detach().numpy())
    test_labels[test_set[i]] = [int(i) for i in flatten(y_test_expected.tolist())]

with open("test_labels.json", "w") as file:
    json.dump(test_labels, file, indent=4)
