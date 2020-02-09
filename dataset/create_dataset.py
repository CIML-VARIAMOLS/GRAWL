import sys
import os
import MDAnalysis as md

def create_verteces(protein_name,file_handle):

	print("verteces computed (", protein_name, ")")

def create_graph(file_handle):
	print("graph created (", protein_name, ")")

# how many folders in raw data? one per protein of interest
ll = os.listdir("./raw_data/")

print("folders in raw_data: ", ll)
# I open a dataset file with the lenght of the dataset of protein
wfile = open(str(len(ll)) + "_graphs_dataset.dat", "w")
# for each prot in the dataset
for fold in ll:
	print(fold)

wfile.close()