# each graph has some verteces and edges
BEGIN_GRAPH
GRAPH_ID (pdb code of the protein)
BEGIN_VERTECES
each atom has the following features:
- ID : atom ID 
- C : is it a carbon atom ? (0/1)
- N : is it a nitrogen atom? (0/1)
- O : is it a oxygen atom? (0/1)
- S : is it a sulphur atom? (0/1)
- HPhob : is it part of a hydrophobic residue? (0/1)
- Amph : is it part of a amphipathic residue? (0/1)
- Pol : is it part of a polar residue? (0/1)
- Ch : is it part of a charged residue? (0/1)
- Bkb : is it part of the protein backbone? (0/1) 
- ...
END_VERTECES
BEGIN_EDGES
# edges added if atoms closer than 1nm (threshold can be changed)
# The graph is undirected 
each edge has the following features:
- first atom index
- second atom index
- inverse of distance (bigger values => closer atoms)
- is there a covalent bond? (0/1)
END_EDGES
END_GRAPH

# for each graph (graph_ID) and M (how many labelled nodes) there will be some mappings (labelings) with the corresponding values of the observable
- file WHATEVER_mappings_$graph_ID$_$M$.txt has the labelings.
example line
[0 1 3 7 .. ] : these are the node IDs which are selected by the labeling procedure
- file WHATEVER_smaps_$graph_ID$_$M$.txt contains the values of the observable
