import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
import argparse

def processData(filename):
    edgelist_file="dataset/"+str(filename)+"/edgelist.txt"
    file=open(edgelist_file,'r')
    content=file.readlines()
    fromNodes=[]
    toNodes=[]
    for line in content:
        a,b=line.split(" ")
        fromNodes.append(int(a))
        toNodes.append(int(b))
    labels_file="dataset/"+str(filename)+"/labels.txt"
    label=open(labels_file,'r')
    label=label.readlines()
    labels=[]
    for line in label:
        a,b=line.split(" ")
        labels.append(int(b))
    labels_matrix=np.zeros((len(labels),max(labels)))
    for i in range(labels_matrix.shape[0]):
        labels_matrix[i,labels[i]-1]=1
    edge_index=torch.tensor([fromNodes,toNodes])
    x=torch.arange(len(labels))
    data=Data(x=x,edge_index=edge_index)
    data.y=labels
    adj_matrix = torch.zeros((data.num_nodes, data.num_nodes))
    adj_matrix[data.edge_index[0], data.edge_index[1]] = 1
    adj_matrix[data.edge_index[1], data.edge_index[0]] = 1
    num=adj_matrix.numpy()
    csc_matrix = sp.csc_matrix(num)
    labels_matrix=sp.csc_matrix(labels_matrix)
    data={
    "network":csc_matrix,
    "group":labels_matrix}
    save_name=str(filename)+".mat"
    sio.savemat(save_name, data)

def main():
    print("start")
    parser = argparse.ArgumentParser(description='Define hyper-parameter')
    parser.add_argument('-f', '--filename', type=str,required=True, help="Relative path to dataset file")
    args = parser.parse_args()
    processData(args.filename)
    print("finish")
if __name__ == "__main__":
    main()