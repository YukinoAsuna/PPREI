'''
Part of the code refers to https://github.com/konsotirop/Invert_Embeddings, which is the official implementation of the paper:
> Sudhanshu Chanpuriya, Cameron Musco, Konstantinos Sotiropoulos, Charalampos E. Tsourakakis. "DeepWalking Backwards: From Embeddings Back to Graphs." ICML 2021.
'''

#import mkl
#mkl.set_num_threads(2)

import os
import pickle

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

# 创建一个日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建一个控制台处理器，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)
class Network:
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
			self.Adjacency = A['A']
		# destroy diag and binarize
		self.Adjacency.setdiag(0)#让对角线元素全变成0
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
		#获取所有不为0元素的位置，并将其设置为1

		# make the graph undirected
		standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)
		'''
		maximum是逐元素比较两个矩阵，并返回一个新矩阵。这个矩阵每个元素都是这两个矩阵该元素的最大值。
		'''

		# select the largest connected component
		_, components = connected_components(standardized_adj_matrix)
		'''
		_是连通分量的个数。
		components是一个list，表示每个点属于哪个连通分量
		例子：
		 >>> graph = [
        ... [0, 1, 1, 0, 0],
        ... [0, 0, 1, 0, 0],
        ... [0, 0, 0, 0, 0],
        ... [0, 0, 0, 0, 1],
        ... [0, 0, 0, 0, 0]
        ... ]
        >>> graph = csr_matrix(graph)
        >>> print(graph)
          (0, 1)	1
          (0, 2)	1
          (1, 2)	1
          (3, 4)	1
    
        >>> n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        >>> n_components
        2
        >>> labels
        array([0, 0, 0, 1, 1], dtype=int32)
		'''
		c_ids, c_counts = np.unique(components, return_counts=True)
		'''
		c_counts可以得到每个连通分量的点的个数
		例子：
		arr = np.array([2, 1, 3, 1, 2, 3, 4, 5, 4, 6])
		unique_elements, counts = np.unique(arr, return_counts=True)
		print(unique_elements)  # Output: [1 2 3 4 5 6]
		print(counts)  # Output: [2 2 2 2 1 1]
		'''

		id_max_component = c_ids[c_counts.argmax()]
		select = components == id_max_component
		standardized_adj_matrix = standardized_adj_matrix[select][:, select]
		'''
		上面一段就是选择属于最大连通分量的点
		'''
		if self.labeled:
			standardized_labels = self.Labels[select]
		else:
			standardized_labels = None

		# remove self-loops
		standardized_adj_matrix = standardized_adj_matrix.tolil()
		standardized_adj_matrix.setdiag(0)
		standardized_adj_matrix = standardized_adj_matrix.tocsr()
		standardized_adj_matrix.eliminate_zeros()
		'''
		上面的操作就是对稀疏矩阵进行存储方式的变换，转为csr这种适合操作的形式
		这样返回Adjacency就是原图的最大连通分量的图，label也是对应的label。
		'''

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

	def netmf_embedding(self, T, skip_max=False):
		"""
		Calculates the NetMF embedding for the network
		Parameters:
			rank (int): Low-rank approximation
			T (int): Optimization Window
		"""
		# Calculate embedding

		n = self.Adjacency.shape[0]
		lap, deg_sqrt = sp.sparse.csgraph.laplacian(self.Adjacency, normed=True, return_diag=True)
		#lap是拉普拉斯矩阵（归一化了），deg_sqrt是度的平方根(也就是对每个点的度取平方根)
		eigen_val, eigen_vec = np.linalg.eigh((sp.sparse.identity(n) - lap).todense())
		#得到拉普拉斯矩阵的特征值和特征向量
		perm = np.argsort(-np.abs(eigen_val))
		#来对特征值的绝对值进行降序排列，并返回对应的索引。
		eigen_val, eigen_vec = eigen_val[perm], eigen_vec[:,perm]
		#按照重要性进行重新排序
		deg_inv_sqrt_diag = sp.sparse.spdiags(1./deg_sqrt, 0, n, n)
		#创建一个对角线矩阵。该对角线矩阵是由1.0 / deg_sqrt数组中的元素组成
		vol = self.Adjacency.sum()
		
		eigen_val_trans = sp.sparse.spdiags(eigen_val[1:] * (1-eigen_val[1:]**T) / (1-eigen_val[1:]), 0, n-1, n-1)
		
		if skip_max:
			self.Embedding =  np.log(1 + vol/T * deg_inv_sqrt_diag @ eigen_vec[:,1:] @ eigen_val_trans @ eigen_vec[:,1:].T @ deg_inv_sqrt_diag)
		else:
			self.Embedding =  np.log(np.maximum(1., 1 + vol/T * deg_inv_sqrt_diag @ eigen_vec[:,1:] @ eigen_val_trans @ eigen_vec[:,1:].T @ deg_inv_sqrt_diag))
		return

	def low_rank_embedding( self, rank ):
		# Low-rank approximation
		w, v = np.linalg.eigh( self.Embedding) 
		order = np.argsort(np.abs(w))[::-1]
		w, v = w[order[:rank]], v[:,order[:rank]]
		self.LR_Embedding = v @ np.diag(w) @ v.T
		
		return

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
class Optimizer:
	def __init__(self, adjacency, embedding, filename, seq_number, rank, device=torch.device("cuda"), dtype=torch.double):
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

	def learnNetwork( self, max_iter=50, method='autoshift' ):
		
		# FUNCTIONS
		def pmi_loss_10_elt_param(elts, n, logit_mode='raw', vol=0., skip_max=False, given_edges=False ):
			elts_tensor = torch.tensor(elts, device=self.device, dtype=self.dtype, requires_grad=True)
			adj_recon = torch.zeros(n,n, device=self.device, dtype=self.dtype)
			#shape就是要优化的图
			#下面的logit_mode是要构建邻接矩阵
			if logit_mode == 'individual':
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = torch.sigmoid(elts_tensor)
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=True, shift=0.)
			elif logit_mode == 'raw':
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = elts_tensor
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=False)
			elif logit_mode == 'softmax':
				adj_recon[np.triu_indices(n,1)] = torch.nn.functional.softmax(elts_tensor, dim=0) * (vol/2)
			elif logit_mode == 'autoshift':
				#构造邻接矩阵的方法是通过自动调整（autoshift）来使得图的总度数接近预设值vol/2
				self.shift = 0.
				for i in range(10):
					self.shift = self.shift - (torch.sigmoid(elts_tensor+self.shift).sum() - (vol/2)) / (torch.sigmoid(elts_tensor+self.shift) * (1. - torch.sigmoid(elts_tensor+self.shift))).sum()
				'''
				self.shift - (torch.sigmoid(elts_tensor+self.shift).sum() - (vol/2)) 
				通过计算差值从而计算新的self.shift值，通过使得图的总度数接近预设值vol/2
				这里sigmoid是为了使得邻接矩阵的取值在0,1之间。
				sigmoid 函数的导数是 sigmoid(x) * (1 - sigmoid(x))。因此通过误差除以导数，从而更新shift的值
				'''
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = torch.sigmoid(elts_tensor+self.shift)
					#通过 sigmoid 函数映射到 (0, 1) 之间，并赋值给 adj_recon 邻接矩阵的上三角部分。
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=True, shift=shift)
        
			adj_recon = adj_recon + adj_recon.T
			deg_recon = adj_recon.sum(dim=0)
			vol_recon = deg_recon.sum()
			#with torch.no_grad():
			#	print( "Adjacency error: ", (adj_recon - self.adjacency).pow(2).sum() / (self.adjacency).pow(2).sum() )
			#	print( "Min Degree:", torch.min( deg_recon ) )
			#	print( "Max Degree:", torch.max( deg_recon ) )
			p_recon = (1. / deg_recon)[:,np.newaxis] * adj_recon
			#(1. / deg_recon)[:,np.newaxis].扩维度的操作。将shape由[1000]变成[1000,1]
			p_recon_2 = p_recon @ p_recon
    
			p_recon_5 = (p_recon_2 @ p_recon_2) @ p_recon
			eyes=torch.eye(n,device=self.device)
			p_geo_series_recon = ( ((p_recon + p_recon_2) @ (eyes + p_recon_2)) + p_recon_5 ) @ (eyes + p_recon_5)
			#这一步应该就是求和
    			
			if skip_max:
				pmi_recon_exact = torch.log((vol_recon/10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis,:])
			else:
				pmi_recon_exact = torch.log(torch.clamp((vol_recon/10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis,:], min=1.))
				#clamp就是类似于max函数了。
			'''
			上面就完成了PPMI矩阵的近似
			'''
			loss_pmi = (pmi_recon_exact - self.pmi).pow(2).sum() / (self.pmi).pow(2).sum()
			loss_deg = (deg_recon - self.deg).pow(2).sum() / self.deg.pow(2).sum()
			loss_vol = (vol_recon - self.vol).pow(2) / (self.vol**2)
    
			loss = loss_pmi
			print('{}. Loss: {}\t PMI: {}\t Vol: {}\t Deg: {:.2f}'.format(self.iter_num, math.sqrt( loss.item() ), math.sqrt( loss_pmi.item() ), loss_vol.item(), loss_deg.item()))
			loss.backward()
			with torch.no_grad():
				#if self.iter_num == 150:
				#	print("Loss: {}, Error: {}".format( loss.item(), (adj_recon - self.adjacency).pow(2).sum() / (self.adjacency).pow(2).sum() ) )
				if torch.isnan(loss):
					pass
					#print("Loss is nan on the following adj, p_sym, pmi:")
					#print(adj_recon)
					#print(p_recon)
					#print(pmi_recon_exact)
					#print(np.linalg.norm(pmi_recon_exact.detach().numpy() - pmi.detach().numpy()) / np.linalg.norm(pmi.detach().numpy()), 
				#      np.linalg.norm(pmi_recon_exact.detach().numpy() - pmi_exact) / np.linalg.norm(pmi_exact))
			gradients = elts_tensor.grad.cpu().numpy().flatten()
			#print("Nan in gradient?", np.argwhere( np.isnan(gradients) ) )
			#print("Gradient norm: ", np.linalg.norm( gradients ))
			return loss.cpu(), gradients

		def callback_elt_param(x_i):
			self.elts = x_i
			self.iter_num += 1
			if self.iter_num % 5 == 0:
				np.save( 'adj_recon/' +  self.filename + '_' + self.rank +'_recon_elts.npy', expit(self.elts + self.shift.detach().cpu().numpy()))
		
		# MAIN OPTIMIZATION
		np.random.seed()
		self.elts = np.random.uniform(0,1, size=(self.n*self.n-self.n) // 2 )
		#是为了计算一个 n × n 的上三角矩阵中不包含对角线元素的元素个数。
		self.iter_num = 0
		self.elts *= 0#就是为了确定一个shape
		res = scipy.optimize.minimize(pmi_loss_10_elt_param, x0=self.elts, 
                              args=(self.n,'autoshift',self.vol, False), jac=True, method='L-BFGS-B',
                             callback=callback_elt_param,
                              tol=np.finfo(float).eps, 
                                  options={'maxiter':max_iter, 'ftol':np.finfo(float).eps, 'gtol':np.finfo(float).eps}
                             )
	"""
	可以用于最小化目标函数，并找到满足约束条件的最优解。
	scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, bounds=None, constraints=(), tol=None, ...)
	fun就是目标函数
	x0: 优化问题的初始猜测。通常是待优化变量的1-D 数组。
	args就是fun的参数
	method: 可选参数，优化算法的名称或算法对象。默认为None，此时将选择一个适合问题的算法。常用的算法包括"BFGS"、"L-BFGS-B"、"SLSQP"等。
	callback:回调函数是在每次优化迭代时被调用的函数，可以用于监视优化过程、记录优化结果或进行其他操作。
	jac:目标函数的梯度
	tol:优化算法的收敛容差。
	"""
	
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def main():
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	
	# Read arguments
	parser = argparse.ArgumentParser(description='Define filename, window, rank and limit')
	parser.add_argument('-f', '--filename', required=True, help="Relative path to dataset file")
	parser.add_argument('-i', '--it', required=False, default="1", help="Number of iteration")
	parser.add_argument('-w', '--window', required=False, default="10", help="Window for SGD (default is 10)")
	parser.add_argument('-r', '--rank', required=False, default="128", help="Rank of approximation for embedding (default is 128)")
	parser.add_argument('-l', '--limit', required=False, default="4", help="Limit the number of threads (default is 4)")
	args = parser.parse_args()
	print(args)
	# Check validity of given arguments
	check_positive( args.it )
	window = check_positive( args.window )
	rank = check_positive( args.rank )
	
	# Limit number of threads
	#thread_limit( args.limit )
	
	# Network instance
	N = Network( )
	N.loadNetwork( args.filename, True )
	N.standardize()
	#N.k_core( 2 )
	#print( "Nodes: ", N.getAdjacency().shape )
	skip_max = False
	#if args.filename == "cora.mat" or args.filename == "citeseer.mat":
	#	skip_max = False 
	N.netmf_embedding( window, skip_max )
	N.low_rank_embedding( rank )
	
	# Learn Adjacency Matrix having same embedding
	P = Optimizer( N.getAdjacency(), N.get_LR_embedding(), args.filename, args.it, args.rank )
	P.learnNetwork() 

if __name__ == "__main__":
    main()
