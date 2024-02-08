
'''
Most codes are from https://github.com/konsotirop/Invert_Embeddings, which is the official implementation of the paper:
        DeepWalking Backwards: From Embeddings Back to Graphs
        Sudhanshu Chanpuriya, Cameron Musco, Konstantinos Sotiropoulos, Charalampos E. Tsourakakis
'''


#import mkl
import os

#mkl.set_num_threads(1)
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pickle
import numpy as np
import scipy as sp
import scipy.sparse, scipy.io, scipy.optimize
from scipy.special import expit
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import torch
from torch import nn
from scipy.sparse import csgraph, linalg, csc_matrix, triu
import random
import pickle as pk
import math
import networkx as nx
from collections import Counter
from scipy.linalg import pinv
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import sys
import warnings
import argparse
import os
import itertools
import random
from random import randint
# Import Network Class
from invert_embeddings import Network, check_positive
from diameter_estimation import startBFS, getDiameter
from operator import itemgetter
from sklearn import preprocessing
import pandas as pd
import pickle as pk

binarize_top = lambda M, num_ones : M >= np.quantile(M, 1 - num_ones / M.size) # set top num_ones to one, else to zero

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
class Reconstructed:
	def __init__( self, number_of_nodes, volume ):
		self.n = number_of_nodes
		self.vol  = volume
		self.Adjacency = None
		self.Adjacency_Binarized = None
		self.G	= None
		self.G_complete = None

	def loadNetwork( self, filename, method="coin_toss", device=torch.device("cpu"), dtype=torch.double):
		'''
		load the PPR embedding matrix
		'''
		elts = np.load( filename )
		elts_tensor = torch.tensor(elts, device=device, dtype=dtype, requires_grad=True)
		adj_recon = torch.zeros(self.n,self.n, device=device, dtype=dtype)
		adj_recon[np.triu_indices(self.n,1)] = elts_tensor
		adj_recon = adj_recon + adj_recon.T
		adj_recon = adj_recon.detach().numpy()
		self.Adjacency_Binarized = np.copy(adj_recon)
		self.Adjacency = scipy.sparse.csc_matrix( adj_recon )
		self.binarize( method ) # Binarize adjacency matrix

	def binarize( self, method="coin_toss" ):
		if method=="add_edge":# DELETE
			# For every node add its highest entry in an effort not to disconnect the graph
			max_row = np.argmax(self.Adjacency_Binarized, axis=1)
			for i in range(len(max_row)):
				self.Adjacency_Binarized[i,max_row[i]] = 1.0
				self.Adjacency_Binarized[max_row[i],i] = 1.0
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( binarize_top(self.Adjacency_Binarized, int(self.vol))).astype('int')
		elif method == "maxst":
			max_st = csgraph.minimum_spanning_tree( -1.*self.Adjacency_Binarized )
			row,col = max_st.nonzero()
			for i in range(len(row)):
				self.Adjacency_Binarized[row[i],col[i]] = 1.0
				self.Adjacency_Binarized[col[i],row[i]] = 1.0
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( binarize_top(self.Adjacency_Binarized, int(self.vol))).astype('int')
		elif method == "threshold":
			max_st = csgraph.minimum_spanning_tree( -1.*self.Adjacency_Binarized )
			row,col = max_st.nonzero()
			for i in range(len(row)):
				self.Adjacency_Binarized[row[i],col[i]] = 1.0
				self.Adjacency_Binarized[col[i],row[i]] = 1.0
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( preprocessing.binarize( self.Adjacency_Binarized, 0.5 ) )
		elif method == "coin_toss2":
			matrix_values = self.Adjacency[np.triu_indices(self.n,1)]
			coin_tosses = np.random.rand(1,self.n*(self.n-1)//2)
			print(matrix_values.shape, coin_tosses.shape)
			outcome = (matrix_values >= coin_tosses).astype(float)
			self.Adjacency_Binarized = np.zeros((self.n,self.n))
			self.Adjacency_Binarized[np.triu_indices(self.n,1)] = outcome
			self.Adjacency_Binarized = self.Adjacency_Binarized + self.Adjacency_Binarized.T
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( self.Adjacency_Binarized )
		elif method == "coin_toss":
			n = self.Adjacency_Binarized.shape[0]
			adj_mst = np.where(sp.sparse.csgraph.minimum_spanning_tree(-1*self.Adjacency_Binarized).todense() != 0, 1., 0.)
			#sp.sparse.csgraph.minimum_spanning_tree. Obtain the minimum spanning tree
			adj_mst += adj_mst.T
			adj_probs = np.copy(self.Adjacency_Binarized)
			adj_probs[adj_mst == 1.] = 1.
			adj_probs[adj_mst == 0.] *= self.Adjacency_Binarized.sum() / adj_probs.sum()
			self.Adjacency_Binarized = 1. * (np.random.rand(*adj_probs.shape) < adj_probs)
			self.Adjacency_Binarized[np.triu_indices(n)] = 0
			self.Adjacency_Binarized += self.Adjacency_Binarized.T
			self.Adjacency_Binarized = sp.sparse.csr_matrix(self.Adjacency_Binarized)
		return

	def low_rank_embedding( self, rank ):
		# Low-rank approximation
		w, v = np.linalg.eigh( self.Embedding )
		order = np.argsort(np.abs(w))[::-1]
		w, v = w[order[:rank]], v[:,order[:rank]]
		self.LR_Embedding = v @ np.diag(w) @ v.T

		return

	def getAdjacency( self ):
		return self.Adjacency

	def getAdjacencyBinarized( self ):
		return self.Adjacency_Binarized

	def setNetworkXGraph( self ):
		self.G = nx.from_scipy_sparse_matrix(self.Adjacency_Binarized)
		self.G.remove_edges_from(nx.selfloop_edges(self.G))

		return

	def getNetworkXGraph( self ):
		if not self.G:
			self.setNetworkXGraph()
		return self.G

	def setNetworkXComplete( self ):
		self.G_complete = nx.from_scipy_sparse_matrix( triu(self.Adjacency, 1) )
		self.G_complete.remove_edges_from(nx.selfloop_edges(self.G_complete))
		
		return

	def getNetworkXComplete( self  ):
		if not self.G_complete:
			self.setNetworkXComplete()
		return self.G_complete
	
	def closenessCentrality( self ):
		cc = nx.closeness_centrality( self.G )
		return cc

	def pageRank( self, binarized=False, num_iterations=100, d=0.85 ):
		if binarized:
			pr_vector = nx.pagerank( self.G )
		else:
			A  = self.getAdjacency()
			deg = A.sum(axis=0)
			deg = np.array( deg )
			deg_inv = np.diag( 1./deg[0] )
			M = deg_inv @  A
			N = M.shape[1]
			pr_vector = np.random.rand(N, 1)
			pr_vector = pr_vector / np.linalg.norm(pr_vector, 1)
			M_hat = (d * M.T + (1 - d) / N)
			for i in range(num_iterations):
				pr_vector = M_hat @ pr_vector
			pr_vector = dict(zip(range(N), pr_vector))
		return pr_vector

def spectralClustering( G, k=2 ):
	# Due to Ng, Jordan, and Weiss (2002)
	Lsym = nx.normalized_laplacian_matrix( G )
	eigvals, eigvecs = linalg.eigsh( Lsym, k = k, which='SM')
	eigvecs = normalize( eigvecs )
	kmeans = KMeans(n_clusters=k, random_state=0).fit( eigvecs)
	return eigvecs, kmeans.labels_

def frobenius_error( adj_org, adj_recon, adj_recon_bin ):
	#print("1. Frobenius Norm")

	#print("Original - Reconstructed")
	recon_frob_error = np.linalg.norm(adj_recon.todense() - adj_org.todense()) / np.linalg.norm(adj_org.todense())

	#print("Original - Binarized")
	bin_frob_error = np.linalg.norm(adj_recon_bin.todense() - adj_org.todense()) / np.linalg.norm(adj_org.todense())
	#print("True: {}, Reconstructed: {}, Binarized: {}".format("N/A", recon_frob_error, bin_frob_error))
	return recon_frob_error, bin_frob_error

def maximum_degree( adj_org, adj_recon, adj_recon_bin ):
	maxd_org = max(adj_org.sum(axis=1)).tolist()
	maxd_rec = max(adj_recon.sum(axis=1)).tolist()
	maxd_rec_bin = max(adj_recon_bin.sum(axis=1)).tolist()
	#print("2. Maximum Degree:")
	#print("True: {}, Reconstructed: {}, Binarized: {}".format(maxd_org[0][0], maxd_rec[0][0], maxd_rec_bin[0][0]))
	return maxd_rec, maxd_rec_bin



# Triangle reconstruction
def triangle_counting( adj ):
	def varied_step(start,stop,change_point, later_step ):
		step = 1
		while start < stop:
			yield start
			start += step
			if start >= change_point:
				step = later_step
			if start >= 500:
				step = 1000

	cuber = lambda t: t ** 3

	def count_triangles(A, step=10):
		n = A.shape[0]
		d = A.sum(axis=1).tolist()
		d = [x[0] for x in d]
		d = np.array([round(x) for x in d])
		x_axis = []
		y_axis = []
		d_sorted = sorted(d)
		d_thres = []
		for i in range(0,len(d_sorted),200):
			d_thres.append(d_sorted[i])
		d_thres.append(d_sorted[-1])
		d_thres = np.unique(d_thres)
		for i in range(0,len(d_thres)):
			mask = d<=d_thres[i]
			idx = np.nonzero( mask )
			Z = A[np.ix_(idx[0],idx[0])]
			eigvals = np.linalg.eigvalsh(Z)
			triangles = np.sum([cuber(x) for x in eigvals])/6
			x_axis.append( d_thres[i] )
			y_axis.append( triangles/n )
		return x_axis, y_axis
	x,y = count_triangles(adj.todense(),25)

	return x,y


def readMatlabCommunities(labels):
	indices = zip(*labels.nonzero())
	partitions = [[] for _ in range(labels.shape[0])]
	for i,j in indices:
		partitions[i].append(j)

	return partitions

def edge_conductance(labels, G_org, G_recon, weight=None):

	# Communities
	partitions = readMatlabCommunities(labels)
	
	# Community sizes
	community_size = defaultdict( int )
	for node in G_org.nodes():
		for com in partitions[node]:
			community_size[com] += 1

	# Conductance per community
	# original graph
	inside_edges = defaultdict(int)
	spanning_edges = defaultdict( int )
	communities = set()
	for u,v in G_org.edges():
		u_communities = partitions[u]
		v_communities = partitions[v]
		common = set([x for x in u_communities if x in v_communities])
		allCommunities = set(u_communities + v_communities ) 
		for c1 in allCommunities:						 # Change this
			communities.add(c1)
			if c1 in common:
				inside_edges[c1] += 2 # No double counting later
			else:
				spanning_edges[c1] += 1  
	org_conductance = dict()
	for c in communities:
		org_conductance[c] = spanning_edges[c] / (spanning_edges[c] + inside_edges[c])
	# reconstructed graph
	inside_edges = defaultdict(int)
	spanning_edges = defaultdict( int )
	communities = set()
	for u,v in G_recon.edges():
		if weight:
			w = G_recon[u][v]['weight']
		else:
			w = 1
		u_communities = partitions[u]
		v_communities = partitions[v]
		common = set([x for x in u_communities if x in v_communities])
		allCommunities = set(u_communities + v_communities )  
		for c1 in allCommunities:						 # Change this
			communities.add(c1)
			if c1 in common:
				inside_edges[c1] += 2*w # No double counting later
			else:
				spanning_edges[c1] += 1*w	  
	recon_conductance = dict()
	for c in communities:
		recon_conductance[c] = spanning_edges[c] / (spanning_edges[c] + inside_edges[c])

	communities = sorted(list(communities))
	score = 0
	#print("(a) Info for all communities")
	for c in communities:
	#	print("Community: {}".format(c))
	#	print("Original Graph: {}, Reconstructed Graph: {}".format(org_conductance[c], recon_conductance[c]))
	#	print("Original Graph - Reconstructed Graph = {}".format(org_conductance[c]-recon_conductance[c]))
		score = score + (org_conductance[c]-recon_conductance[c])*(community_size[c]/len(partitions))
	#print("Total score: {}".format(score))
	#print("(b) 10 largest communities")
	top10 = dict(sorted(community_size.items(), key = itemgetter(1), reverse = True)[:max(10,len(communities))])
	for k,v in top10.items():
		pass
		#print("Community: {} (size: {}), True Network: {}, Reconstructed Network: {}".format(k,v, org_conductance[k], recon_conductance[k]))
	#print("(c) Best community")
	com, con = sorted(org_conductance.items(), key = itemgetter(1), reverse = False)[:1][0]
	#print("Community: {} (size: {}), True Network: {}, Reconstructed Network: {}".format(com, community_size[com], con, recon_conductance[com]))
	return org_conductance, recon_conductance, community_size

def nodeTriangles( A ):
	l, u = np.linalg.eigh( A.todense() )
	l = l ** 3
	u = np.array( u )
	u = u ** 2
	triangles = ( l @ u.T ) / 2
	return triangles

def fiedlerEigenvalue( A ):
	L =  csgraph.laplacian(A, normed=False)
	eigvals = linalg.eigsh( L, k = 2, which='SM', return_eigenvectors=False )
	return eigvals[0]

def overlap(tupleA, tupleB, k=10):
	A_index = [s[0] for s in tupleA[:k]]
	B_index = [s[0] for s in tupleB[:k]]
	common = [x for x in A_index if x in B_index]
	return len(common)

def gencoordinates(m, n, total_pairs):
	seen = set()
	pairs = []
	cnt = 0
	x, y = randint(m, n), randint(m, n)
	while cnt < total_pairs:
		seen.add((x, y))
		pairs.append( (x, y) )
		cnt += 1
		x, y = randint(m, n), randint(m, n)
		while (x, y) in seen:
			x, y = randint(m, n), randint(m, n)
	return pairs

def average_paths( G, pairs, weight=None ):
	length = 0
	found = 0
	for p in pairs:
		u,v = p
		try:
			l = nx.shortest_path_length(G, source=u, target=v, weight=weight)
			found += 1
		except:
			# No path found
			l = 0
		length += l
	return length / found

def collectStatistics( filename, rank,method="coin_toss" ):

	if not sys.warnoptions:
		warnings.simplefilter("ignore")

	
	# Original Network
	Original = Network()
	Original.loadNetwork( filename, True )
	Original.standardize()
	#Original.k_core(2)
	number_of_nodes = Original.getAdjacency().shape[0]
	number_of_edges = Original.getAdjacency().sum() / 2
	#print("Nodes: {}, Edges: {}".format(number_of_nodes, number_of_edges))
	maxd_org = max(Original.getAdjacency().sum(axis=1)).tolist()

	# Diameter
	org_diameter, org_effective = getDiameter(  Original.getNetworkXGraph(), 500 )

	# All pairs shortest path
	pairs_count = int( number_of_nodes *  math.log(number_of_nodes,2))
	pairs = gencoordinates(0, number_of_nodes-1, pairs_count )
	avg_pl = average_paths( Original.getNetworkXGraph(), pairs, None )

	# Gather results from true network to a dictionary
	results = dict()
	results['true'] = dict()
	results['true']['frobenius'] = 0
	results['true']['max_degree'] = maxd_org
	results['true']['diameter'] = org_diameter
	results['true']['apl'] = avg_pl
	#results['true']['conductance'] = INSIDE FOR LOOP
	results['true']['nodes'] = number_of_nodes
	results['true']['edges'] = number_of_edges
	#print("Real network stats:")
	#print(results['true'])
	# Reconstructed Networks
	for iteration in [rank]:
		# Initiate dictionaries to gather results
		results['rank'+str(iteration)] = dict() # Expected network
		results['rank'+str(iteration)+'bin'] = dict()
		#print(" ---------- ")
		#print("Rank: {}".format(iteration))
		#print(" ---------- ") 
		it_cnt = str(iteration)
		filename = os.path.splitext( filename )[0]
		folder = 'adj_recon/'
		reconstructed_network = folder +  filename + '_PPR_' + it_cnt +'_recon_elts.npy'
		Recon = Reconstructed( *Original.getNodesVolume() )
		Recon.loadNetwork( reconstructed_network, method )
		Recon.setNetworkXGraph()
		results['rank'+str(iteration)]['edges'] = Recon.getAdjacency().sum() / 2
		results['rank'+str(iteration)+'bin']['edges'] = Recon.getAdjacencyBinarized().sum() / 2
		
		# 1. Frobenius Norm Error
		frob_full, frob_bin = frobenius_error( Original.getAdjacency(), Recon.getAdjacency(), Recon.getAdjacencyBinarized() )
		results['rank'+str(iteration)]['frobenius'] = frob_full
		results['rank'+str(iteration)+'bin']['frobenius'] = frob_bin

		# 2. Maximum Degree
		maxd_full, maxd_bin = maximum_degree( Original.getAdjacency(), Recon.getAdjacency(), Recon.getAdjacencyBinarized() )
		results['rank'+str(iteration)]['max_degree'] = maxd_full
		results['rank'+str(iteration)+'bin']['max_degree'] = maxd_bin

		# 3. Diameter
		recon_diameter, recon_effective = getDiameter(  Recon.getNetworkXGraph(), 500 )
		results['rank'+str(iteration)]['diameter'] = None
		results['rank'+str(iteration)+'bin']['diameter']  = recon_diameter

		# 4. Average path length
		avg_pl_bin = average_paths( Recon.getNetworkXGraph(), pairs, None )
		results['rank'+str(iteration)]['apl'] = None
		results['rank'+str(iteration)+'bin']['apl'] = avg_pl_bin

		# 5. Edge Conductance
		if Original.isLabeled():
			pass
		# 	#print("5. Conductance")
		# 	#print("-----	i) Binarized Network   -------")
		if Original.isLabeled():
			org_con, bin_con, comsize = edge_conductance( Original.getLabels(), Original.getNetworkXGraph(), Recon.getNetworkXGraph() )
		if Original.isLabeled():
			org_con, recon_con, comsize = edge_conductance( Original.getLabels(), Original.getNetworkXGraph(), Recon.getNetworkXComplete(), weight='weight' )
		results['true']['conductance'] = org_con
		results['true']['com_sizes'] = comsize
		results['rank'+str(iteration)]['conductance'] = recon_con
		results['rank'+str(iteration)+'bin']['conductance'] = bin_con

		if 0 == 1: # Do not run those yet
			#print("Average path length (Complete)")
			avg_pl_compl = average_paths( Recon.getNetworkXComplete(), pairs, 'weight' )
			#print("Original Graph: {:3f}, Reconstructed Graph (binarized): {:3f}".format(avg_pl, avg_pl_compl))
		
			recon_complete_assort = assortativity( Recon.getNetworkXComplete(), 'weight' )
			#print("3. Assortativity:")
			#print("Original Graph: {:3f}, Reconstructed Graph (Complete): {:3f}".format(org_assort, recon_complete_assort))

			#print("Clustering Coefficient")
			#print( "Original: {}".format(nx.average_clustering(  Original.getNetworkXGraph() )))
			#print( "Reconstructed (complete): {}".format(nx.average_clustering(  Recon.getNetworkXComplete() )))
		#pk.dump( results, open( 'pickleFolder/' + filename + method + '_results.pk', "wb" ) )
	return results

def main():
	parser = argparse.ArgumentParser(description='Define filename, window, rank and limit')
	parser.add_argument('-f', '--filename', required=True, help="Relative path to dataset file")
	parser.add_argument('-r', '--rank', required=True, help="the rank of reduced dimensional representation")

	args = parser.parse_args()
	folder_name = "result"
	create_folder(folder_name)
	recon_network_stats=collectStatistics(args.filename,args.rank)
	Error = []
	Path_origin = []
	Path_recon = []
	origin = []
	recon = []
	Index = []
	rank=args.rank

	print("start")
	print("* Relative Adjacency Error: {}".format(recon_network_stats['rank' + str(rank)]['frobenius']))

	print("* Average Path Length, Original: {}, Reconstructed: {}".format(recon_network_stats['true']['apl'],
																		  recon_network_stats[
																			  'rank' + str(rank) + 'bin']['apl']))
	print("* Conductance of Ground-Truth Communities")
	for community, conductance in recon_network_stats['true']['conductance'].items():
		#print("Community Index: {}".format(community))
		Error.append(recon_network_stats['rank' + str(rank)]['frobenius'])
		Path_origin.append(recon_network_stats['true']['apl'])
		Path_recon.append(recon_network_stats['rank' + str(rank) + 'bin']['apl'])
		Index.append(community)
		recon_conductance = recon_network_stats['rank' + str(rank)]['conductance'][community]
		origin.append(conductance)
		recon.append(recon_conductance)
		# recon_conductance = recon_network_stats['rank' + str(rank)]['conductance'][community]
		# print("Conductance(True Network): {}, Conductance(Reconstructed Network): {}".format(conductance,
		# 																					 recon_conductance))
	metric = pd.DataFrame({
		"Error": Error,
		"Path_origin": Path_origin,
		"Path_recon": Path_recon,
		'Community Index': Index,
		'origin conductance': origin,
		'recon conductance': recon
	})
	current_t = time.gmtime()
	network_name=args.filename.split('.')[0]
	metric.to_csv(f'result/result_' + network_name + '_PPR_' + str(rank)+"_"+time.strftime('%Y-%m-%d-%H', current_t) + '.csv')
	print("finish")		
if __name__ == "__main__":
	main()
