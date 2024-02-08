'''
Most codes are from https://github.com/konsotirop/Invert_Embeddings, which is the official implementation of the paper:
        DeepWalking Backwards: From Embeddings Back to Graphs
        Sudhanshu Chanpuriya, Cameron Musco, Konstantinos Sotiropoulos, Charalampos E. Tsourakakis
'''


#import mkl
#mkl.set_num_threads(2)

import os
import pickle
from tqdm import tqdm
import pandas as pd
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import scipy as sp
import scipy.sparse, scipy.io, scipy.optimize
from scipy.special import expit
from scipy import sparse, stats
import torch
from torch import nn
from scipy.sparse import csgraph, linalg, csc_matrix
import random
import pickle as pk
import math
from scipy.sparse.csgraph import connected_components
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import argparse
import warnings
import sys
import networkx as nx
from collections import Counter
import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
import os

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")



class Network1:
	def __init__( self ):
		self.Adjacency = None
		self.Labels = None
		self.Embedding = None
		self.LR_Embedding = None
		self.G = None
		self.labeled = False
		self.rank = None

	def loadNetwork( self, network_filename, binarize=False ):
		"""
		Loads a network and its labels (if available)
		"""
		A=scipy.io.loadmat( network_filename )
		try:
			self.Adjacency = A['network']
		except:
			self.Adjacency = A['net']
		# destroy diag and binarize
		self.Adjacency.setdiag(0)
		if binarize:
			self.Adjacency.data = 1. * (self.Adjacency.data > 0)
    
		# Load labels - if available
		try:
			self.Labels = A['group']
			try:
				self.Labels = self.Labels.todense().astype(np.int32)
				self.Labels = np.array( self.Labels )

			except:
				self.Labels = self.Labels.astype(np.int32)
			self.labeled = True
		except:
			self.Labels = [None]
			self.labeled = False
		#print("Labeled?", self.labeled)
		return

	def SBM( self, sizes, probs ):
		self.G = nx.stochastic_block_model( sizes, probs )
		self.Adjacency = nx.to_scipy_sparse_matrix(self.G)
		self.Labels = []
		[self.Labels.extend([i for _ in range(sizes[i])]) for i in range(len(sizes))]
		self.Labels = np.array( self.Labels )
		self.labeled = True
		return

	def standardize( self ):
		"""
		Make the graph undirected and select only the nodes
		belonging to the largest connected component.

		:param adj_matrix: sp.spmatrix
			Sparse adjacency matrix
		:param labels: array-like, shape [n]

		:return:
			standardized_adj_matrix: sp.spmatrix
			Standardized sparse adjacency matrix.
			standardized_labels: array-like, shape [?]
			Labels for the selected nodes.
		"""
		# copy the input
		standardized_adj_matrix = self.Adjacency.copy()

		# make the graph unweighted
		standardized_adj_matrix[standardized_adj_matrix != 0] = 1

		# make the graph undirected
		standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)

		# select the largest connected component
		_, components = connected_components(standardized_adj_matrix)
		c_ids, c_counts = np.unique(components, return_counts=True)

		id_max_component = c_ids[c_counts.argmax()]
		select = components == id_max_component
		standardized_adj_matrix = standardized_adj_matrix[select][:, select]
		if self.labeled:
			standardized_labels = self.Labels[select]
		else:
			standardized_labels = None

		# remove self-loops
		standardized_adj_matrix = standardized_adj_matrix.tolil()
		standardized_adj_matrix.setdiag(0)
		standardized_adj_matrix = standardized_adj_matrix.tocsr()
		standardized_adj_matrix.eliminate_zeros()

		self.Adjacency, self.Labels = standardized_adj_matrix, standardized_labels

		return
	def k_core( self, k ):
		"""
		Keeps the k-core of the input graph
		:param k: int 
		"""
		self.setNetworkXGraph()
		core_numbers = nx.core_number( self.G )
		select = [key for key,v in core_numbers.items() if v >=k]
		self.Adjacency = self.Adjacency[select][:, select]
		self.Labels =  self.Labels[select]
		
	def PPR_embedding(self,T=10,alpha=0.1,epi=1e-6):
		A = self.getAdjacency().todense()
		deg = np.sum(A, axis=1)
		P = A / deg
		P_values = dict()
		for i in range(T+1):
			P_values[i] = np.linalg.matrix_power(P, i)
		pai = np.zeros_like(P)
		for i in range(T+1):
			pai += alpha * (1 - alpha) ** i * P_values[i]
		pai=pai/epi
		pai[pai<1]=1
		pai=np.log(pai)
		self.PPR=pai
		return

	def low_rank_embedding( self, rank,matrix):
		# Low-rank approximation
		w, v = np.linalg.eigh( matrix )
		order = np.argsort(np.abs(w))[::-1]
		w, v = w[order[:rank]], v[:,order[:rank]]
		self.LR_Embedding = v @ np.diag(w) @ v.T
		
		return self.LR_Embedding

	def closenessCentrality( self ):
		cc = nx.closeness_centrality( self.G )
		return cc
	
	def pageRank( self ):
		pr_vector = nx.pagerank( self.G )
		return pr_vector

	def getAdjacency( self ):
		return self.Adjacency

	def get_LR_embedding( self ):
		return self.LR_Embedding

	def setNetworkXGraph( self ):
		self.G = nx.from_scipy_sparse_matrix(self.Adjacency)
		self.G.remove_edges_from(nx.selfloop_edges(self.G))
		
		return
		
	def getNetworkXGraph( self ):
		if not self.G:
			self.setNetworkXGraph()
		return self.G

	def getNodesVolume( self ):
		return self.Adjacency.shape[0], np.array(self.Adjacency.sum(axis=1)).flatten().sum()

	def getLabels( self ):
		return self.Labels
	
	def isLabeled( self ):
		return self.labeled
class Optimizer1:
	def __init__(self, adjacency, embedding, filename, seq_number, rank, device=torch.device("cpu"), dtype=torch.double):
		self.n = adjacency.shape[0]
		self.device = device
		self.dtype = dtype
		self.pmi = torch.tensor(embedding, device=self.device, dtype=self.dtype, requires_grad=False)
		deg = np.array(adjacency.sum(axis=1)).flatten()
		self.vol = deg.sum()#迹
		self.deg = torch.tensor(deg, device=self.device, dtype=self.dtype, requires_grad=False)
		self.filename = os.path.splitext( filename )[0]
		self.sequence = seq_number
		if not os.path.exists(self.filename + '_networks'):
   	 		os.makedirs(self.filename + '_networks')
		self.folder = self.filename + '_networks/'
		self.rank = rank
		self.shift = 0.
		self.adjacency = torch.tensor( adjacency.todense(), device=self.device, dtype=self.dtype, requires_grad=False)
	# def PPR_learnNetwork(self, max_iter=50, method='autoshift',vol=0., T=15,alpha=0.1,epi=1e-5):
	# 	iter_num = 0
	# # FUNCTIONS
	# 	elts_tensor = torch.zeros(((self.n * self.n - self.n) // 2), device=self.device, dtype=self.dtype, requires_grad=True)
	# 	adj_recon = torch.zeros(self.n, self.n, device=self.device, dtype=self.dtype)
	# 	optim=torch.optim.Adam([elts_tensor],lr=0.001)
	# 	for i in tqdm(range( max_iter)):
	# 		self.shift = 0.
	# 		optim.zero_grad()
	# 		for i in range(10):
	# 			self.shift = self.shift - (torch.sigmoid(elts_tensor + self.shift).sum() - (vol / 2)) / (
	# 					torch.sigmoid(elts_tensor + self.shift) * (
	# 					1. - torch.sigmoid(elts_tensor + self.shift))).sum()
	# 		adj_recon[np.triu_indices(self.n, 1)] = torch.sigmoid(elts_tensor + self.shift)
	#
	# 		adj_recon = adj_recon + adj_recon.T
	# 		deg_recon = torch.sum(adj_recon,dim=1)
	# 		vol_recon = deg_recon.sum()
	# 		P=adj_recon/deg_recon
	# 		P_values=dict()
	# 		for i in range(T+1):
	# 			P_values[i]=torch.linalg.matrix_power(P,i)
	# 		pai=torch.zeros_like(P)
	# 		for i in range(T+1):
	# 			pai+=alpha*(1-alpha)**i*P_values[i]
	# 		P=pai/epi
	# 		P[P<1]=1
	# 		ppr_recon_exact=torch.log(P)
	#
	# 		loss_pmi = (ppr_recon_exact - self.pmi).pow(2).sum() / (self.pmi).pow(2).sum()
	# 		loss_deg = (deg_recon - self.deg).pow(2).sum() / self.deg.pow(2).sum()
	# 		loss_vol = (vol_recon - self.vol).pow(2) / (self.vol ** 2)
	#
	# 		loss = loss_pmi
	# 		print('{}. Loss: {}\t PMI: {}\t Vol: {}\t Deg: {:.2f}'.format(iter_num, math.sqrt(loss.item()),
	# 																		math.sqrt(loss_pmi.item()),
	# 																		loss_vol.item(),
	# 																		loss_deg.item()))
	# 		iter_num+=1
	# 		loss.backward()
	# 		optim.step()


	def PPMI_learnNetwork( self, max_iter=50, method='autoshift' ):

		# FUNCTIONS
		def pmi_loss_10_elt_param(elts, n, logit_mode='raw', vol=0., skip_max=False, given_edges=False):
			elts_tensor = torch.tensor(elts, device=self.device, dtype=self.dtype, requires_grad=True)
			adj_recon = torch.zeros(n, n, device=self.device, dtype=self.dtype)

			if logit_mode == 'individual':
				if not given_edges:
					adj_recon[np.triu_indices(n, 1)] = torch.sigmoid(elts_tensor)
				else:
					adj_recon[np.triu_indices(n, 1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges,
																		  activation=True, shift=0.)
			elif logit_mode == 'raw':
				if not given_edges:
					adj_recon[np.triu_indices(n, 1)] = elts_tensor
				else:
					adj_recon[np.triu_indices(n, 1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges,
																		  activation=False)
			elif logit_mode == 'softmax':
				adj_recon[np.triu_indices(n, 1)] = torch.nn.functional.softmax(elts_tensor, dim=0) * (vol / 2)
			elif logit_mode == 'autoshift':
				self.shift = 0.
				for i in range(10):
					self.shift = self.shift - (torch.sigmoid(elts_tensor + self.shift).sum() - (vol / 2)) / (
								torch.sigmoid(elts_tensor + self.shift) * (
									1. - torch.sigmoid(elts_tensor + self.shift))).sum()
				if not given_edges:
					adj_recon[np.triu_indices(n, 1)] = torch.sigmoid(elts_tensor + self.shift)
				else:
					adj_recon[np.triu_indices(n, 1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges,
																		  activation=True, shift=shift)

			adj_recon = adj_recon + adj_recon.T
			deg_recon = adj_recon.sum(dim=0)
			vol_recon = deg_recon.sum()
			# with torch.no_grad():
			#	print( "Adjacency error: ", (adj_recon - self.adjacency).pow(2).sum() / (self.adjacency).pow(2).sum() )
			#	print( "Min Degree:", torch.min( deg_recon ) )
			#	print( "Max Degree:", torch.max( deg_recon ) )
			p_recon = (1. / deg_recon)[:, np.newaxis] * adj_recon
			p_recon_2 = p_recon @ p_recon

			p_recon_5 = (p_recon_2 @ p_recon_2) @ p_recon
			p_geo_series_recon = (((p_recon + p_recon_2) @ (torch.eye(n) + p_recon_2)) + p_recon_5) @ (
						torch.eye(n) + p_recon_5)

			if skip_max:
				pmi_recon_exact = torch.log((vol_recon / 10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis, :])
			else:
				pmi_recon_exact = torch.log(
					torch.clamp((vol_recon / 10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis, :], min=1.))
			loss_pmi = (pmi_recon_exact - self.pmi).pow(2).sum() / (self.pmi).pow(2).sum()
			loss_deg = (deg_recon - self.deg).pow(2).sum() / self.deg.pow(2).sum()
			loss_vol = (vol_recon - self.vol).pow(2) / (self.vol ** 2)

			loss = loss_pmi
			print('{}. Loss: {}\t PMI: {}\t Vol: {}\t Deg: {:.2f}'.format(self.iter_num, math.sqrt(loss.item()),
																		  math.sqrt(loss_pmi.item()), loss_vol.item(),
																		  loss_deg.item()))
			loss.backward()
			with torch.no_grad():
				# if self.iter_num == 150:
				#	print("Loss: {}, Error: {}".format( loss.item(), (adj_recon - self.adjacency).pow(2).sum() / (self.adjacency).pow(2).sum() ) )
				if torch.isnan(loss):
					pass
			gradients = elts_tensor.grad.numpy().flatten()
			print(gradients)

			return loss, gradients

		def callback_elt_param(x_i):
			self.elts = x_i
			self.iter_num += 1
			if self.iter_num % 5 == 0:
				np.save( 'adj_recon/' +  self.filename + '_' + self.rank +'_recon_elts.npy', expit(self.elts + self.shift.detach().numpy()))
		
		# MAIN OPTIMIZATION
		np.random.seed()
		self.elts = np.random.uniform(0,1, size=(self.n*self.n-self.n) // 2 )
		self.iter_num = 0
		self.elts *= 0
		res = scipy.optimize.minimize(pmi_loss_10_elt_param, x0=self.elts,
                              args=(self.n,'autoshift',self.vol, False), jac=True, method='L-BFGS-B',
                             callback=callback_elt_param,
                              tol=np.finfo(float).eps, 
                                  options={'maxiter':max_iter, 'ftol':np.finfo(float).eps, 'gtol':np.finfo(float).eps}
                             )
	
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue
pmi_loss=[]
gradient_list=[]
vol_loss=[]
deg_loss=[]
iter_num=[]

iter=0
def PPr_loss(elts,PPR,n,filename,vol,T,alpha,epi):
	global iter
	elts_tensor = torch.tensor(elts, dtype=torch.double, requires_grad=True)
	adj_recon = torch.zeros(n, n,  dtype=torch.double)
	shift = 0.
	for i in range(10):
		shift = shift - (torch.sigmoid(elts_tensor + shift).sum() - (vol / 2)) / (
				torch.sigmoid(elts_tensor + shift) * (
				1. - torch.sigmoid(elts_tensor + shift))).sum()

	adj_recon[np.triu_indices(n, 1)] = torch.sigmoid(elts_tensor + shift)
	adj_recon = adj_recon + adj_recon.T
	deg_recon = torch.sum(adj_recon,dim=1)
	vol_recon = deg_recon.sum()
	P=adj_recon/deg_recon
	pai=torch.zeros_like(P)
	for i in range(T+1):
		pai+=alpha*(1-alpha)**i*torch.linalg.matrix_power(P,i)
	P=pai/epi
	P=torch.clamp(P, min=1)
	ppr_recon_exact=torch.log(P)
	loss_pmi = (ppr_recon_exact - PPR).pow(2).sum() / (PPR).pow(2).sum()
	loss_vol = (vol_recon - vol).pow(2) / (vol ** 2)
	loss_pmi.backward()
	gradients = elts_tensor.grad.numpy().flatten()
	elts1=elts_tensor.detach().numpy()
	print('{}. Loss: {}\t  Vol: {}\t'.format(iter, math.sqrt(loss_pmi.item()),
			loss_vol.item()))
	iter_num.append(iter)
	gradient_list.append(gradients)
	pmi_loss.append(loss_pmi.item())
	vol_loss.append(loss_vol.item())
	iter+=1
	np.save('adj_recon/' + filename + '_PPR_128'  + '_recon_elts.npy',
			expit(elts1 + shift.detach().numpy()))
	return loss_pmi, gradients
def main():
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	folder_name = "adj_recon"
	create_folder(folder_name)
	# Read arguments
	parser = argparse.ArgumentParser(description='Define hyper-parameter')
	parser.add_argument('-f', '--filename', type=str,required=True, help="Relative path to dataset file")
	parser.add_argument('-i', '--i', required=False, type=int,default="40", help="Number of iteration")
	parser.add_argument('-t', '--t', required=False, type=int,default="10", help="Number of step")
	parser.add_argument('-a', '--alpha', required=False, type=float,default="0.1", help="Number of iteration")
	parser.add_argument('-e', '--epi', required=False, type=float,default="1e-7", help="Number of iteration")
	parser.add_argument('-r', '--rank', required=False, type=int,default="128", help="Rank of approximation for embedding (default is 128)")
	args = parser.parse_args()
	print(args)
	# Check validity of given arguments
	check_positive( args.i )
	rank = check_positive( args.rank )

	# Network instance
	N = Network1( )
	# Load network
	print("loading")
	N.loadNetwork( args.filename, True )
	# Get largest connected component
	N.standardize()
	# PPREI matrix
	skip_max = False
	N.PPR_embedding(T=args.t,epi=args.epi,alpha=args.alpha)
	# And low-rank dimensional embedding
	PPR=N.low_rank_embedding( rank,N.PPR )
	PPR=torch.tensor(PPR)
	n=N.getAdjacency().shape[0]
	filename=args.filename.split('.')[0]
	max_iter=args.i
	np.random.seed()
	elts = np.random.uniform(0,1, size=(n*n-n) // 2 )
	iter_num = 0
	elts *= 0
	vol=N.getAdjacency().sum()
	print("optimizing")
	res = scipy.optimize.minimize(PPr_loss, x0=elts,
                                args=(PPR, n,filename,vol,args.t,args.alpha,args.epi), jac=True, method='L-BFGS-B',
                                tol=np.finfo(float).eps,
                                options={'maxiter': max_iter}
                                )
	print("finish")


if __name__ == "__main__":
    main()
