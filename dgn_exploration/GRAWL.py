# Import necessary libraries
# Deep learning modules
import json
import random
from pathlib import Path
import pickle

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import reset

# Wand Landau modules
import numpy as np
import datetime as dt
import math
import os
import sys

print ("Initial time: ", dt.datetime.now())

def system_parameters_setup(parfile):
    # receives the name of the parameters file in input
    # parses it and gives back a dictionary
    ###################################################
    print("reading parameters file named ", parfile)
    parameters = {}
    pars = open(parfile,"r")
    lines = pars.read().split("\n")
    for ln in lines[:-1]:
        split_list = ln.split()
        if len(split_list) != 2:
            raise Exception(f"badly formatted parameter line\n{ln}")
        else:
            par_name = split_list[0]
            par_value = split_list[1]
            parameters[par_name] = par_value
            print("parameter ", par_name, " = ", par_value)
    pars.close()
    return parameters

print("loaded packages")
print("torch.cuda.is_available() = ", torch.cuda.is_available())
if torch.cuda.is_available():
    #map_location = torch.device("cuda:1")
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
print("map_location" , map_location)

# Define the Model

def convert_mapping(mapping):
    print("conv_mapping: ",torch.nonzero(mapping, as_tuple=True))

def append_mapping(mapping,smap_mapping,savebin,min_norm_visited,delnorm):
    #routine that appends mapping to a file/creates file if empty
    filename = "Mappings_norm_%lf-%lf.dat"%(min_norm_visited + delnorm*savebin, min_norm_visited + delnorm*(savebin+1))
    # check for existence
    if filename in os.listdir("./Mappings_saved_4ake/"):
        filename = os.path.join("./Mappings_saved_4ake/",filename)
        wfile=open(filename,"a")
    else:
        print("filename", filename," not present, creating file...")
        filename = os.path.join("./Mappings_saved_4ake/",filename)
        wfile=open(filename,"w")
    # check mapping
    correct_map = numpy.count_nonzero(mapping == 1)
    if correct_map != 214:
      raise Exception("mapping", mapping, "has N != 214")
    # writing file
    for k in range(len(mapping)):
        if mapping[k] == 1:
            wfile.write("%d "%k)
    wfile.write(" - Smap: %lf\n"%smap_mapping)
    wfile.close()

class TrentoConv(MessagePassing):
    def __init__(self, nn, aggregation='mean', eps=0.1, **kwargs):
        super(TrentoConv, self).__init__(aggr=aggregation, **kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr):
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr[:, 0].unsqueeze(1)


class TrentoDGN(nn.Module):
    # class Deep Graph Network
    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        batchnorm = config.get('batchnorm', True)
        
        self.predictor = predictor_class(dim_node_features=dim_embedding*num_layers, dim_edge_features=dim_edge_features, dim_target=dim_target, config=config)

        w = torch.empty(2,1)
        torch.nn.init.xavier_uniform_(w)
        self.w = torch.nn.Parameter(w)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else dim_embedding
            if i == 0:
                if batchnorm:
                    conv = Sequential(Linear(dim_input, dim_embedding), BatchNorm1d(dim_embedding), ReLU())
                else:
                    conv = Sequential(Linear(dim_input, dim_embedding), ReLU())
            else:
                if batchnorm:
                    conv = TrentoConv(Sequential(Linear(dim_input, dim_embedding), BatchNorm1d(dim_embedding), ReLU()), aggregation=config['neighbor_aggregation'])
                else:
                    conv = TrentoConv(Sequential(Linear(dim_input, dim_embedding), ReLU()), aggregation=config['neighbor_aggregation'])
            self.layers.append(conv)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # remove the beta feature
        x = torch.cat((x[:, :9], x[:, -1].unsqueeze(1)), dim=1)

        # Get a weight w_0 or w_1 depending on whether the node has been selected
        w_x = torch.index_select(self.w, dim=0, index=x[:,-1].long())

        x_all = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            else:
                x = layer(x, edge_index, edge_attr)
            x_all.append(x)

        x = torch.cat(x_all, dim=1) * w_x
        data.x = x

        return self.predictor(data)

    
class LinearTrentoGraphPredictor(nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super(LinearTrentoGraphPredictor, self).__init__()
        if config['readout'] == 'sum':
            self.readout = global_add_pool
        elif config['readout'] == 'mean':
            self.readout = global_mean_pool
        self.final_act = nn.Identity()

        self.out = nn.Linear(dim_node_features, dim_target)

    def forward(self, data):
        node_x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.readout(node_x, batch)
        return self.final_act(self.out(x))

# Load checkpoint

def load_checkpoint(root_folder, dataset):
    """ 
    load the model 
    checkpoint and json should be correctly formatted:
    - {dataset}_checkpoint.pth 
    - {dataset}_best_config.json
    """
    dim_node_features, dim_edge_features, dim_target = 10, 1, 1
    model_state_dict = torch.load(Path(root_folder, f'{dataset}_checkpoint.pth'),map_location=map_location)
    with open(Path(root_folder, f'{dataset}_best_config.json'), 'r') as f:
        config = json.load(f)

    model = TrentoDGN(dim_node_features, dim_edge_features, dim_target, LinearTrentoGraphPredictor, config['best_config']['config']['supervised_config'])
    model.load_state_dict(model_state_dict['model_state'])
    model.eval()
    return model

def load_graph_data(features_file, adjacency_file):
    """
    Parse the input and returns node features, edge indices and edge attributes

    :returns: a torch.tensor of shape (no_nodes, no_features), 
            a torch.tensor of size (2, no_edges) with the adjacency information,
            and a torch.tensor of size (no_edges, edge_features) with the edge attributes
    """    
    # Creo feature e adj random per demo
    x = torch.rand(100, 9)
    edge_index, _ = torch_geometric.utils.dense_to_sparse((torch.rand(100, 100) > 0.8).int())
    edge_attr = torch.rand(edge_index.shape[1], 2)
    
    return x, edge_index, edge_attr
    

def add_sites(x, sites):
    """
    Adds the list of sites to the graph
    :param x: The torch.tensor features matrix of size (no_nodes, 9)
    :param sites: The torch.tensor mapping column of size (no_nodes, 1)
    Return: The complete torch.tensor features matrix of size (no_nodes, 10)
    """
    if sites is None:
        # Aggiungo un mapping random, in attesa che Marco passi quello vero come un 
        #random_sites = (torch.rand(x.shape[0]) > 0.5).float()
        #x[:, -1] = random_sites
        raise Exception("None sites passed to add_sites")
    else:
        x[:, -1] = sites
    return x

def create_graph_object(x, edge_index, edge_attr):
    return Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)])

# main body

parameters = system_parameters_setup(sys.argv[1])
print(parameters)
dataset = parameters["dataset"] # '6d93' or '4ake'
task = parameters["task"] # 'test' for testing, 'bench' for benchmarking, 'run' for production runs

# load graph 

with open(f'./graph_{dataset}.pkl', 'rb') as f:
    x, edge_index, edge_attr = pickle.load(f)
x, edge_index, edge_attr = torch.tensor(x), torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr)

# load checkpoint
model = load_checkpoint(root_folder='.', dataset=dataset)

# Create input graph

#for i in range(len(opt_maps)):
#    first_opt_binary = torch.zeros(1656, dtype=torch.float)
#    for el in opt_maps[i]:
#      first_opt_binary[el] = 1.0
#    x = add_sites(x, first_opt_binary)
#    g = create_graph_object(x, edge_index, edge_attr)
#    model = model.to('cpu')
#    g = g.to('cpu')
#    
#    #K = 0.0013363595837560089  # Used to conform with previous work from Giulini et al.
#    smap = model(g).detach().item()
#    print(f'S_map is {smap}')

###########################################

print("starting wang landau exploration")

n_heavy = int(parameters["n_heavy"])
n_rand = int(parameters["n_rand"])
at_nr=np.zeros(shape=(n_heavy), dtype=int)
#
if task != "test":
    # no testing. build directories
    os.mkdir(f'Histograms_checks_{dataset}')
    os.mkdir(f'Histograms_final_{dataset}')
    #os.mkdir(f'Min_maps_checks_{dataset}')
    #os.mkdir(f'Min_maps_final_{dataset}')
    #os.mkdir(f'Max_maps_checks_{dataset}')
    #os.mkdir(f'Max_maps_final_{dataset}')
#
check_histo=int(parameters["check_histo"])
min_mc_moves=int(parameters["min_mc_moves"])
#
mapping = torch.zeros(n_heavy, dtype=torch.int)
mapping_prime = torch.zeros(n_heavy, dtype=torch.int)
#
## cgsites
ncg=int(parameters["ncg"])
print(f"Number of CG sites = {ncg}")
# mapping

rd_sel = np.random.choice(np.arange(0,n_heavy), size=ncg, replace=False)
#print(rd_sel)
#print("shape rd_sel", rd_sel.shape)
for el in rd_sel:
    mapping[el] = 1

print("first mapping")
convert_mapping(mapping)
# atom_ret and atom_nnret
atom_ret = np.nonzero(mapping)
print("shape atom_ret", atom_ret.shape)
#print(atom_ret)
atom_nnret = np.nonzero(mapping == 0)
print("shape atom_nnret", atom_nnret.shape)

max_norm = float(parameters["max_norm"])
#ref_maxnorm = 108.5 # highest value in ref histogram: print all values higher than this or lower than minnorm
min_norm = float(parameters["min_norm"])
delx= float(parameters["delx"]) # bin width
del_smap = float(parameters["del_smap"]) # delta parameter for saving mappings
nbins=int((max_norm)/delx)+1
print("maxnorm - minnorm - delx - nbins", max_norm, min_norm,delx,nbins)
#Counter to start saving maps when all bins have been visited 
MC_COUNT_GIU=0
#Variable saying every how many MC sweeps mappings have to be saved
MC_sweep_save=1
# array containing the range of visited bins that are taken as reference
ref_visited_bins=np.zeros(shape=(nbins),dtype=int)
#
#histogram for the wang-landau sampling
histo=np.zeros(shape=(nbins))
#log of density of states for the wang-landau sampling
log_dofstates=np.zeros(shape=(nbins))
#initial value of f factor for wang langau implementation ---> here it does not change!
logf=float(parameters["logf"])
logf_final=float(parameters["logf_final"])
#
##save first mapping norm and associated bin
##mapping_prime=metric_matrix.dot(mapping)
##norm_mapping=numpy.dot(mapping,mapping_prime)/neighat
#
###########################################################
###########################################################
x = add_sites(x,  mapping)
g = create_graph_object(x, edge_index, edge_attr)
# model is loaded to selected locationù
if torch.cuda.is_available():
    gpu1 = torch.device("cuda:0")
    model = model.to(gpu1)
    print("type g", type(g))
    g = g.to(gpu1)
else:
    model = model.to(map_location)
    g = g.to(map_location)
## infer!
smap = model(g).detach().item()
print(f'starting S_map is {smap}')
###########################################################
###########################################################
ibin=int(smap/delx)
# creating a temporary mapping by cloning torch mapping
mapping_temp = mapping.clone()
## THEORETICALLY WE SHOULD START OUR SIM FROM THE CENTER OF THE INTERVAL...WHO CARES??
##flatness criterion
pflat=float(parameters["pflat"])
#this is the first time you run it
#first=True
#
##Array counting if a certain bin has been visited on the fly during the wang-landau, cheched against the reference
visited_bins=np.zeros(shape=(nbins),dtype=int)
#
##saying that the first guy was visited
log_dofstates[ibin]+=logf
visited_bins[ibin]=1
#this tells us when a new bin is visited
visited=True
#this tells us when all bins with respect to the reference have been visited to start saving the mappings
visited_all=False
#Check that the total number of mappings saved is the right one, in that case stop
all_maps_saved=False
print (f"Beginning Wang-Landau mapping space exploration to save mappings: {dt.datetime.now()}")
#
tot_knt = 0

""" 
functions for wang landau:
    - make_a_move
    - perform_inference
    - print_final_histogram
"""

def make_a_move(torch_mapping, np_retained, np_not_retained):
    """
    routine that creates mapping_temp by changing a site

    """
    at1_idx = np.random.randint(0,ncg)
    at1 = np_retained[at1_idx]
    at2_idx = np.random.randint(0,n_heavy-ncg)
    at2 = np_not_retained[at2_idx]
    torch_mapping[at1] = 0
    torch_mapping[at2] = 1
    #print(f"at1 {at1}, at2 {at2}")
    return torch_mapping,at1_idx,at2_idx

def perform_inference(x,edge_index,edge_attr, mapping_temp, model):
    """
    loads the model and performs inference
    There's an if on the device used
    """
    x = add_sites(x, mapping_temp)
    g = create_graph_object(x, edge_index, edge_attr)
    if torch.cuda.is_available():
        gpu1 = torch.device("cuda:0")
        model = model.to(gpu1)
        #print("type g", type(g))
        g = g.to(gpu1)
    else:
        model = model.to(map_location)
        g = g.to(map_location)
    # infer!
    smap_temp = model(g).detach().item()
    return smap_temp

def print_final_histogram(histo,log_dofstates,logf,nbins,delx,count,final):
    """
    printing intermediate and final histograms
    """
    if final == "yes":
        print("log_dofstates when FLATNESS CONDITION SATISFIED", log_dofstates)
        print("Iteration with factor %.10lf completed, histogram flattened"%logf,dt.datetime.now())
        outhist=open("Histograms_final_4ake/Histo_final_f_%.10lf.dat"%logf,"w")
    else:
        print("Checking histo of f = %.10lf, MC_move = %d, checkbox = %d"%(logf,mc_move,check))
        outhist=open("Histograms_checks_4ake/Histo_%.10lf_%d_%d.dat"%(logf,mc_move,check),"w")
    for wbin in range(nbins):
        xc=delx*wbin
        outhist.write("%lf %lf %lf\n"%(xc,histo[wbin]/count,log_dofstates[wbin]))
    outhist.close()

if task == "runvi":
##while all_maps_saved==False:
    print(task)
elif task == "bench":
    # benchmarking
    bench_steps = 10000
    print("benchmarking with 10000 inferences")
    t_start = dt.datetime.now()
    for s in range(bench_steps):
        # do inference 
        smap_bench = perform_inference(x,edge_index,edge_attr, mapping, model)
        print(f"smap_bench[{s}] = {smap_bench}")
    t_end = dt.datetime.now() - t_start
    print(f"time required to do inference on {bench_steps} = {t_end}")

elif task == "test":
    #for n in range(200):     
    #    print("n = ",n)       
    while logf>=logf_final:
    ##reset all counters
        if (visited==True):
            print("Iteration of Wang-Landau with f=%.10lf"%logf)
        count=0
        mc_move=0
        check=check_histo+1
        check_flat=False
        #reset histogram
        histo=np.zeros(shape=(nbins))
        #while check_flat==False:
        for s in range(n_rand):
            print("count ", count)
            print("tot_knt", tot_knt)
            # sweep of n_rand steps
            for dummy in range(n_rand):
                visited=True
                #print("shapes: ",atom_ret.shape, atom_nnret.shape)
                mapping_temp,at1_idx,at2_idx = make_a_move(mapping_temp,atom_ret,atom_nnret)
                # swapping last item of the arrays with the selected atom, it's O(1)
                at1, at2 =  atom_ret[at1_idx].clone(), atom_nnret[at2_idx].clone()
                #print(f"at1,{at1}, at2 {at2}")
                atom_ret[at1_idx] = atom_ret[-1]
                atom_nnret[at2_idx] = atom_nnret[-1]
                #print(f"atom_ret[{at1_idx}] {atom_ret[at1_idx]} atom_nnret[{at2_idx}] {atom_nnret[at2_idx]}")
                atom_ret[-1] = at1
                atom_nnret[-1] = at2
                #print(f"atom_ret[-1] {atom_ret[-1]} atom_nnret[-1] {atom_nnret[-1]}")
                #convert_mapping(mapping_temp)
                ##########################################################
                ##########################################################
                smap_temp = perform_inference(x,edge_index,edge_attr,mapping_temp,model)
                if smap_temp > max_norm or smap_temp < min_norm:
                    print(f'out of the box S_map_temp is {smap_temp}')
                tot_knt += 1
                ##########################################################
                ##########################################################
                if (smap_temp>min_norm)and(smap_temp<max_norm): #the proposed mapping is in the window, run the acceptance
                    ibin_new=int(smap_temp/delx)
                    prob=log_dofstates[ibin]-log_dofstates[ibin_new]
                    #acceptance rule
                    if(np.log(np.random.uniform()) < prob): #move accepted, update mapping and stuff
                        ibin=ibin_new
                        smap=smap_temp 
                        mapping[at1]=0
                        mapping[at2]=1
                        # updating atom_ret,atom_nnret
                        atom_ret[-1] = at2
                        atom_nnret[-1] = at1
                        #print("atom_ret", str(atom_ret))
                        #print("atom_nnret", str(atom_nnret))
                        if visited_bins[ibin]==0:
                           print("new visited_bin %d with norm %lf"%(ibin,delx*ibin))
                           print(f"visitor smap {smap}")
                           print("visitor mapping")
                           convert_mapping(mapping)
                           visited_bins[ibin]=1
                           visited=False
                           minlogdof=np.min(log_dofstates[np.nonzero(log_dofstates)])
                    # rejection
                    else:
                        mapping_temp[at1]=1
                        mapping_temp[at2]=0
                    # update histogram and density of states    
                    histo[ibin]+=1
                    count+=1
                    if visited==False:
                        print("visited == False, ibin ", ibin)
                        log_dofstates[ibin]+=minlogdof
                        break
                    else:
                       log_dofstates[ibin]+=logf 
                # condition on min_explorable and max_explorable
                else:    #BINDER STUFF: the proposed mapping is outside the window, so reject the move and update histo and dos of the old mapping
                    print("out of bounds smap, rejecting and updating current bin")
                    mapping_temp[at1]=1
                    mapping_temp[at2]=0
                    histo[ibin]+=1
                    count+=1
                    log_dofstates[ibin]+=logf
            #update counters
            mc_move+=1
            check+=1
            if mc_move%10==0: 
                print("MCmove: %d"%(mc_move))   
                print("log_dofstates", log_dofstates) 
            # save mapping if logf factor is already < 0.5
            if logf < 0.5:
                print("saving mapping")
                #savebin = int((smap-min_explorable)/0.775)
                int((smap-min_norm)/1.0)
                print("saving mapping in savebin", savebin)
                convert_mapping(mapping)
                append_mapping(mapping,smap,savebin,min_norm,del_smap) # 1.0 is the delnorm, so that we have 19 bins for the mapping
            #Everything now is checked provided that this is not a newly visited bin                
            #a bin has not been visited, restart the histogram
            if (visited==False):
                check_flat=True
                break
            #all bin have bin visited check histogram depending on the number of moves
            elif(check>=check_histo)and(visited==True)and(mc_move>=min_mc_moves):
                print_final_histogram(histo,log_dofstates,logf,nbins,delx,count,check,final="no")
                #reset the check flag
                check=0
                # compute_histo_average(visited_bins)
                count_bins = np.nonzero(visited_bins)
                avg_hist = np.mean(histo[count_bins])
                print("Average of histogram: %lf, on a total number of bins= %d"%(avg_hist,len(count_bins)))
                #For an histogram to be flat, the logdos of all visited states must be higher than pflat the average.
                checkall=True
                for bins in range(nbins):
                    if visited_bins[bins]==1:
                        if (histo[bins]<pflat*avg_hist)or(histo[bins]>((2.-pflat)*avg_hist)):
                            checkall=False
                            break
                if (checkall==False):
                    print("Flatness condition not satisfied: some bins have a number of counts smaller than pflat*average = %lf or higher than (2-pflat)*average = %lf"%(pflat*avg_hist,(2.-pflat)*avg_hist))
                else:
                    print("FLATNESS CONDITION SATISFIED")
                    print_final_histogram(histo,log_dofstates,logf,nbins,delx,count,check,final="yes")       
                    #the histogram is flat: update the logf and check_flat
                    check_flat=True
                    logf=logf/2.
    print ("Final time: ", dt.datetime.now())    