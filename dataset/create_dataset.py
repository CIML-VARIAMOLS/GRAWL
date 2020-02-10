import sys
import os
import mdtraj as mdt
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np

def create_edges(path_to_pdb,file_handle):
	file_handle.write("BEGIN_EDGES\n")
	# reading pdb
	prot = mdt.load_pdb(path_to_pdb)
	no_h = prot.topology.select("type != H")
	prot = mdt.load_pdb(path_to_pdb, atom_indices=no_h)
	# bonds
	table, bonds = prot.topology.to_dataframe()	
	# cycling for distances
	array_of_dist = cdist(prot.xyz[0], prot.xyz[0])
	for n in range(array_of_dist.shape[0]):
		# 1 nm of cutoff distance
		neighs = [s for s in range(n+1,n+len(array_of_dist[n,n:])) if array_of_dist[n,s] < 1]
		# looking for bonds:
		pattern = np.where(bonds[:,0] == float(n))
		bonded_atoms = bonds[pattern,1]
		for nn in neighs:
			if nn in bonded_atoms:
				# 1 meaning covalent bond
				file_handle.write(str(n)+"\t"+str(nn)+"\t"+str(round(0.1/array_of_dist[n,nn],4))+"\t1\n")
			else:
				file_handle.write(str(n)+"\t"+str(nn)+"\t"+str(round(0.1/array_of_dist[n,nn],4))+"\t0\n")
	print("verteces computed (", path_to_pdb.split("/")[-1], ")")
	file_handle.write("END_EDGES\n")
	file_handle.write("END_GRAPH\n")



def create_graph(path_to_pdb,file_handle,graph_ID):
	file_handle.write("BEGIN_GRAPH\n")
	file_handle.write("graph_ID : "+ graph_ID + "\n")
	file_handle.write("BEGIN_VERTECES\n")
	file_handle.write("ID\tC\tN\tO\tS\tHPhob\tAmph\tPol\tCh\tBkb\n")
	# reading pdb
	file = open(path_to_pdb, "r")
	numrows = len(file.read().split("\n"))
	file.close()
	pdb_df = pd.read_csv(path_to_pdb,skiprows = 2,nrows = numrows-4, header=None,sep="\s+")
	print("pdb read! shape ", pdb_df.shape)
	# definition of residues
	hydrophobic_res = ["ALA", "ILE", "LEU", "PHE", "VAL", "PRO", "GLY"]
	amphipathic_res = ["TRP", "TYR", "MET"]
	polar_res = ["GLN", "ASN", "HIS", "SER", "THR", "CYS"]
	charged_res = ["ARG", "LYS", "ASP", "GLU"]
	# writing lines
	atom_ID = 0
	for n in range(pdb_df.shape[0]):
		# check it's not an hydrogen
		if pdb_df.iloc[n,12].startswith("H") == False:
			# 1 is part of the backbone?
			bkb = "0"
			if pdb_df.iloc[n,2] in ["N","C","CA","O"]:
				bkb = "1"
			# 2 atom type
			carbon = oxy = sulph = nitro = "0"
			if pdb_df.iloc[n,12].startswith("C"):
				carbon = "1"
			elif pdb_df.iloc[n,12].startswith("N"):
				nitro = "1"
			elif pdb_df.iloc[n,12].startswith("S"):
				sulph = "1"
			elif pdb_df.iloc[n,12].startswith("O"):
				oxy = "1"
			# 3 restype
			h_phob = amphi = polar = charged = "0"
			if pdb_df.iloc[n,3] in hydrophobic_res:
				h_phob = "1"
			elif pdb_df.iloc[n,3] in amphipathic_res:
				amphi = "1"
			elif pdb_df.iloc[n,3] in polar_res:
				polar = "1"
			elif pdb_df.iloc[n,3] in charged_res:
				charged = "1"
			file_handle.write(str(atom_ID) + "\t"+ carbon + "\t" + nitro + "\t" + oxy + "\t"+ sulph + 
				"\t" + h_phob + "\t" + amphi + "\t" + polar + "\t" + charged + "\t" + bkb + "\n")
			# 4 increment atom index
			atom_ID += 1 
	print("graph created (", path_to_pdb.split("/")[-1], ")")
	file_handle.write("END_VERTECES\n")

# how many folders in raw data? one per protein of interest
ll = os.listdir("./raw_data/")

print("folders in raw_data: ", ll)
# I open a dataset file with the lenght of the dataset of protein
wfile = open(str(len(ll)) + "_graphs_dataset.dat", "w")
# for each prot in the dataset
for fold in ll:
	print("protein ", fold.split("/")[-1])
	protein_folder = os.path.join("./raw_data/",fold)
	print("protein_folder ", protein_folder)
	list_of_pdb = [el for el in os.listdir(protein_folder) if el.endswith(".pdb")]
	print("pdb files: ", list_of_pdb)
	# # get covalent bonds
	# #covalent_bonds = get_bonds(os.path.join(protein_folder,list_of_pdb[0]))
	# print("covalent_bonds ", covalent_bonds)
	# create_graph (and associate bonds to correct indices)
	create_graph(os.path.join(protein_folder,list_of_pdb[0]), wfile, fold.split("/")[-1])
	# create edges
	create_edges(os.path.join(protein_folder,list_of_pdb[0]), wfile)

wfile.close()