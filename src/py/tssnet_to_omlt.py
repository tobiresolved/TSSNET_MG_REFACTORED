# convert the trained neural network to an OMLT model
# then optimizing the model combine with the first stage to find the optimal unit's states
# by Ying Yang, done: 2024-06-06

# import the necessary packages
import json
import torch
import tempfile # to create temporary files
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import torch.onnx
import random

import pyomo.environ as pyo

# omlt
from omlt import OmltBlock
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation, \
FullSpaceSmoothNNFormulation, ReducedSpaceSmoothNNFormulation, ReluBigMFormulation,\
ReluComplementarityFormulation, ReluPartitionFormulation
from omlt.neuralnet.activations import ComplementarityReLUActivation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds

# define the neural network
class Net_ReLU(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(Net_ReLU, self).__init__()
        self.hidden = torch.nn.Linear(n_input, 100)
        self.hidden2 = torch.nn.Linear(100, 60)
        self.hidden3 = torch.nn.Linear(60, 30)
        self.hidden4 = torch.nn.Linear(30, 15)
        self.hidden5 = torch.nn.Linear(15, 4)
        self.predict = torch.nn.Linear(4, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.predict(x)
        return x

# load the data
with open("data/caseA_gel.json", "r") as f:
    gel_data = json.load(f)
with open("results/opt_case_A.json", "r") as f: 
    rel2_data = json.load(f)
with open("results/ram_case_A.json", "r") as f:
    rel1_data = json.load(f)
# read some constants from the system settings
S = gel_data["S"]
T = gel_data["T"]
G = gel_data["G"]
W = gel_data["W"]
D = gel_data["D"]
Chot = gel_data["Chot"]
Ccold = gel_data["Ccold"]
Ton = gel_data["Ton"]
Toff = gel_data["Toff"]
Tcold = gel_data["Tcold"]
T0 = gel_data["T0"]
U0 = gel_data["U0"]
L0 = gel_data["L0"]
x0 = gel_data["x0"]

IN1 = rel1_data["IN"]
IN2 = rel2_data["IN"]
IN = IN1 + IN2

# define the tensors for the data
Pw1 = torch.zeros(IN1, W*T, 1)
ld1 = torch.zeros(IN1, D*T, 1)
opt_u1 = torch.zeros(IN1, G*T, 1)
for it in range(IN1):
    Pw1[it, :, :] = torch.tensor(rel1_data[f"Pw{it+1}"]).view(-1, 1)
    ld1[it, :, :] = torch.tensor(rel1_data[f"load{it+1}"]).view(-1, 1)
    opt_u1[it, :, :] = torch.tensor(rel1_data[f"opt_u{it+1}"]).view(-1, 1)

Pw2 = torch.zeros(IN2, W*T, 1)
ld2 = torch.zeros(IN2, D*T, 1)
opt_u2 = torch.zeros(IN2, G*T, 1)
for it in range(IN2):
    Pw2[it, :, :] = torch.tensor(rel2_data[f"Pw{it+1}"]).view(-1, 1)
    ld2[it, :, :] = torch.tensor(rel2_data[f"load{it+1}"]).view(-1, 1)
    opt_u2[it, :, :] = torch.tensor(rel2_data[f"opt_u{it+1}"]).view(-1, 1)

opt_u = torch.cat((opt_u2, opt_u1), dim=0)
Pw = torch.cat((Pw2, Pw1), dim=0)
ld = torch.cat((ld2, ld1), dim=0)
x_tmp = torch.cat((opt_u.view(IN,G*T), Pw.view(IN,W*T)), dim=1)
x_tmp = torch.cat((x_tmp, ld.view(IN,D*T)), dim=1)
x = Variable(x_tmp)

result_y = []

for yy in [2,3,4]:
    u_test = np.zeros((G, T))
    Pw_test = torch.tensor(rel2_data[f"Pw{yy}"]).view(W, T)
    for i in range(G+1):
        for j in range(T+1):
            u_test[i-1, j-1] = rel2_data[f"opt_u{1}"][j-1][i-1]
    # u_test = torch.tensor(rel2_data[f"opt_u{2}"]).view(G, T)
    y_test = torch.tensor(rel2_data[f"scd_obj{1}"])
    ld_test = torch.tensor(rel2_data[f"load{31}"]).view(D, T)

    fstf_n = G*T # the number of nnt's first input neurons 
    scdf_n = W*T # the number of nnt's second input neurons
    ld_n = D*T # the number of nnt's third input neurons
    n_output = 1 # the number of nnt's output neurons
    n_input = W*T + G*T + D*T # the number of nnt's input neurons

    # opt_u = opt_u.view(1, -1)
    Pw_mx = Pw.view(IN,W*T)
    ld_mx = ld.view(IN,D*T)

    # load trained neural network
    my_nn = Net_ReLU(n_input, n_output) # create a new neural network with same structure
    # my_nn.load_state_dict(torch.load('my_omlt/networks/net_params_6bus.pkl')) # load the trained parameters
    my_nn.load_state_dict(torch.load('models/net_params_caseA.pkl')) # load the trained parameters

    # create the input bounds. in this problem, we know that the first input is between 0 and 1
    input_bounds={}
    for i in range(fstf_n):
        input_bounds[i] = (0, 1)
    for i in range(scdf_n):
        input_bounds[fstf_n+i] = (0, Pw_mx[:,i].max().item())
    for i in range(ld_n):
        input_bounds[fstf_n+scdf_n+i] = (0, ld_mx[:,i].max().item())

    # omlt cannot directly use the pytorch model, so we need to export it to onnx format
    # take first input as x_t
    x_t = x[1].view(1, -1)
    # export the torch model to onnx(Open Neural Network Exchange) format
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            my_nn,
            x_t,
            f,
            input_names=['input'], # specify the name of the input tensor in the onnx model
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        write_onnx_model_with_bounds(f.name, None, input_bounds) # write the onnx model to the temporary file
        print(f"Wrote PyTorch model to {f.name}")
        network_definition = load_onnx_neural_network_with_bounds(f.name) # load the onnx model

    network_definition.scaled_input_bounds

    # create a pyomo model
    model_full = pyo.ConcreteModel() # create a pyomo model

    # define decision variables
    # binary variables
    model_full.s = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.Binary, bounds=(0, 1), initialize=0)
    model_full.d = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.Binary, bounds=(0, 1), initialize=0)
    model_full.u = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.Binary, bounds=(0, 1))

    # continuous variables
    model_full.y = pyo.Var(initialize = 0)
    model_full.Chs = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.NonNegativeReals)
    model_full.Ces = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.NonNegativeReals)
    model_full.Pw = pyo.Var(pyo.RangeSet(W), pyo.RangeSet(T), domain=pyo.NonNegativeReals)
    model_full.ld = pyo.Var(pyo.RangeSet(D), pyo.RangeSet(T), domain=pyo.NonNegativeReals)

    # create an OmltBlock, which is a container for the neural network
    model_full.nn = OmltBlock()
    # convert the neural network to a mathematical model
    formulation_full = ReluBigMFormulation(network_definition)
    # formulation_full = ReluPartitionFormulation(network_definition)
    # formulation_full = ReluComplementarityFormulation(network_definition)

    model_full.nn.build_formulation(formulation_full)

    # define the system constraints
    # define the hot start constraint
    def constraint_hs_rule(mdl, i, t):
        return mdl.Chs[i, t] == Chot[i-1] * mdl.s[i, t]

    # define the excess start constraint
    def constraint_es_rule(mdl, i, t, taoi, fit):
        return mdl.Ces[i, t] >= (Ccold[i-1] - Chot[i-1]) * (mdl.s[i, t] - sum(mdl.d[i, k] for k in range(taoi, t)) - fit)

    # define the initial state constraint
    def constraint_initial_state_rule(mdl, i, t):
        return mdl.u[i, t] == x0[i-1]

    # define the start state constraint
    def constraint_start_state_rule(mdl, i, t, omiga_on):
        return sum(mdl.s[i, k] for k in range(omiga_on, t+1)) <= mdl.u[i, t]

    # define the shut down state constraint
    def constraint_shut_down_state_rule(mdl, i, t, omiga_off):
        return sum(mdl.d[i, k] for k in range(omiga_off, t+1)) <= 1 - mdl.u[i, t]

    # define the units state balance constraint
    def constraint_units_state_balance_rule(mdl, i, t):
        if t == 1:
            return mdl.s[i, t] - mdl.d[i, t] == mdl.u[i, t] - x0[i-1]
        else:
            return mdl.s[i, t] - mdl.d[i, t] == mdl.u[i, t] - mdl.u[i, t-1]
        
    # define the x constraints
    def constraint_x_rule(mdl, i, t):
        return mdl.u[i, t] == mdl.nn.inputs[(i-1)*T + t - 1]

    # define the y constraints
    def constraint_y_rule(mdl):
        return mdl.y == mdl.nn.outputs[0]

    # define the Pw constraints
    def constraint_Pw_rule(mdl, w, t):
        return mdl.Pw[w, t] == mdl.nn.inputs[G*T + (w-1)*T + t - 1]

    # connect Pw to the wind power scenario
    def constraint_PwSce_rule(mdl, w, t):
        return mdl.Pw[w, t] == Pw_test[w-1, t-1].item()

    # define the load constraints
    def constraint_ld_rule(mdl, d, t):
        return mdl.ld[d, t] == mdl.nn.inputs[G*T + W*T + (d-1)*T + t - 1]

    # connect the load to the neural network
    def constraint_ldSce_rule(mdl, d, t):
        return mdl.ld[d, t] == ld_test[d-1, t-1].item()

    # construct the system constraints
    for i in range(1, G+1): # index from 0 for python list, but from 1 for pyomo variable vector
        for t in range(1, T+1):
            # calculate tao and f
            taoi = int(max(1, t-Toff[i-1]-Tcold[i-1]))
            if (t-taoi <= 0) and (max(0, -T0[i-1]) < abs(t-taoi) + 1):
                fit = 1
            else:
                fit = 0
            #calculate omiga_on and omiga_off
            omiga_on = max(0, t-Ton[i-1]) + 1
            omiga_off = max(0, t-Toff[i-1]) + 1
            # construct the hot start constraint
            model_full.add_component(f'constraint_hs_{i}_{t}', pyo.Constraint(rule = constraint_hs_rule(mdl=model_full, i=i, t=t)))
            # # construct the excess start constraint
            model_full.add_component(f'constraint_es_{i}_{t}', pyo.Constraint(rule = constraint_es_rule(mdl=model_full, i=i, t=t, taoi=taoi, fit=fit)))
            # # # construct the initial state constraint
            if t >=1 and t <= U0[i-1]+L0[i-1]: # if t in [1, U0+L0] add the initial state constraint
                model_full.add_component(f'constraint_initial_state_{i}_{t}', pyo.Constraint(rule = constraint_initial_state_rule(model_full, i, t)))
            # if omiga_on <= t and t >= 1+U0[i-1]: # construct the start state constraint
                model_full.add_component(f'constraint_start_state_{i}_{t}', pyo.Constraint(rule = constraint_start_state_rule(model_full, i, t, omiga_on)))
            if omiga_off <= t and t >= 1+L0[i-1]: # construct the shut down state constraint
                model_full.add_component(f'constraint_shut_down_state_{i}_{t}', pyo.Constraint(rule = constraint_shut_down_state_rule(model_full, i, t, omiga_off)))
            # # construct the units state balance constraint
            model_full.add_component(f'constraint_units_state_balance_{i}_{t}', pyo.Constraint(rule = constraint_units_state_balance_rule(model_full, i, t)))
            
            # connect pyomo variables to the neural network
            # construct the x constraints
            model_full.add_component(f'connect_inputs_{i}_{t}', pyo.Constraint(rule = constraint_x_rule(model_full, i, t)))

    # construct the Pw constraints
    for w in range(1, W+1):
        for t in range(1, T+1):
            model_full.add_component(f'constraint_Pw_{w}_{t}', pyo.Constraint(rule = constraint_Pw_rule(model_full, w, t)))
            model_full.add_component(f'constraint_PwSce_{w}_{t}', pyo.Constraint(rule = constraint_PwSce_rule(model_full, w, t)))

    for d in range(1, D+1):
        for t in range(1, T+1):
            model_full.add_component(f'constraint_ld_{d}_{t}', pyo.Constraint(rule = constraint_ld_rule(model_full, d, t)))
            model_full.add_component(f'constraint_ldSce_{d}_{t}', pyo.Constraint(rule = constraint_ldSce_rule(model_full, d, t)))

    # construct the y constraints, just one constraint, so should not be in the loop
    model_full.add_component(f'connect_outputs', pyo.Constraint(rule = constraint_y_rule(model_full)))

    # # print the constraints to verify
    # for i in range(1, G+1):
    #     for t in range(1, T+1):
    # #         print(f'constraint_hs_{i}_{t}:', model_full.component(f'constraint_hs_{i}_{t}').expr)
    # #         print(f'constraint_es_{i}_{t}:', model_full.component(f'constraint_es_{i}_{t}').expr)
    #         # if t >=1 and t <= U0[i-1]+L0[i-1]:
    #         #     print(f'constraint_initial_state_{i}_{t}:', model_full.component(f'constraint_initial_state_{i}_{t}').expr)
    #         print(f'constraint_start_state_{i}_{t}:', model_full.component(f'constraint_start_state_{i}_{t}').expr)
    #         # print(f'constraint_shut_down_state_{i}_{t}:', model_full.component(f'constraint_shut_down_state_{i}_{t}').expr)
    #         # print(f'constraint_units_state_balance_{i}_{t}:', model_full.component(f'constraint_units_state_balance_{i}_{t}').expr)
    #         # print(f'connect_inputs_{i}_{t}:', model_full.component(f'connect_inputs_{i}_{t}'))
    #         # print(f'connect_outputs:', model_full.component(f'connect_outputs'))

    # for i in range(1, W+1):
    #     for t in range(1, T+1):
    #         print(f'constraint_Pw_{i}_{t}:', model_full.component(f'constraint_Pw_{i}_{t}').expr)
    #         print(f'constraint_PwSce_{i}_{t}:', model_full.component(f'constraint_PwSce_{i}_{t}').expr)


    # define the objective function
    model_full.obj = pyo.Objective(expr = (model_full.y) + sum(model_full.Chs[:, :]) + sum(model_full.Ces[:, :]), sense=pyo.minimize)

    # solve the model and query the solution
    status_full = pyo.SolverFactory('gurobi').solve(model_full, tee=False)

    u_rlt = np.zeros((G, T))
    s_rlt = np.zeros((G, T))
    d_rlt = np.zeros((G, T))
    w_rlt = np.zeros((W, T))
    for j in range(1, W+1):
        for t in range(1, T+1):
            w_rlt[j-1, t-1] = pyo.value(model_full.Pw[j,t])
    for i in range(1, G+1):
        for t in range(1, T+1):
            u_rlt[i-1, t-1] = int(pyo.value(model_full.u[i,t]))
            s_rlt[i-1, t-1] = int(pyo.value(model_full.s[i,t]))
            d_rlt[i-1, t-1] = int(pyo.value(model_full.d[i,t]))
    y_results = (pyo.value(model_full.y))
    fst_obj = sum(pyo.value(model_full.Chs[:,:])) + sum(pyo.value(model_full.Ces[:,:]))

    # # concatenate u and w as test_t
    # test_t = torch.tensor(np.concatenate((u_rlt.flatten(), w_rlt.flatten())))
    # test_t = test_t.type(torch.FloatTensor)

    # print out model size and solution values
    # print("# of variables: ", model_full.nn.num_variables())
    # print("# of constraints: ", model_full.nn.num_constraints())

    print("u_nn = ", u_rlt)
    # print("s_nn = ", s_rlt)
    print("u_test = ", u_test)

    print("y_nn = ", y_results)
    print("y_test = ", y_test)
    # add y_results to result_y
    result_y.append(y_results)
    # print("solve time = ", status_full["Solver"][0]["Time"])

print(result_y)