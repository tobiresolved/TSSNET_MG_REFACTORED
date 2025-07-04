# Unified script for OMLT NN-Optimization (Cases A, B, C)
import json
import torch
import tempfile
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.onnx

import pyomo.environ as pyo

from omlt import OmltBlock
from omlt.neuralnet import ReluBigMFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds

import sys

# ---- Case Configs ----
CASE_CONFIGS = {
    "A": {
        "gel_file": "data/caseA_gel.json",
        "opt_file": "results/opt_case_A.json_mean.json",
        "ram_file": "results/ram_case_A.json_mean.json",
        "nn_params": 'models/net_params_caseA.pkl',
        "arch": [100, 60, 30, 15, 4],  # Hidden layer sizes
        "test_idxs": [2, 3, 4],
        "ld_test_idx": 5,
        "pw_test_idx": 1,
        "scd_test_idx": 1
    },
    "B": {
        "gel_file": "data/caseB_gel.json",
        "opt_file": "results/opt_case_B.json",
        "ram_file": "results/ram_case_B.json",
        "nn_params": 'models/net_params_caseB.pkl',
        "arch": [180, 120, 60, 30, 15, 4],
        "test_idxs": [1, 2, 3, 4, 5],
        "ld_test_idx": 6,
        "pw_test_idx": 1,
        "scd_test_idx": 1
    },
    "C": {
        "gel_file": "data/caseC_gel.json",
        "opt_file": "data/opt_results_caseC.json",
        "ram_file": "data/ram_results_caseC.json",
        "nn_params": 'my_omlt/networks/net_params_caseC.pkl',
        "arch": [360, 180, 90, 45, 20, 10],
        "test_idxs": [1, 2, 3, 4, 5],
        "ld_test_idx": 11,
        "pw_test_idx": 1,
        "scd_test_idx": 1
    }
}

# ---- Parse case ----
case = sys.argv[1] if len(sys.argv) > 1 else "A"
conf = CASE_CONFIGS[case]

# ---- Dynamic NN Definition ----
class Net_ReLU(torch.nn.Module):
    def __init__(self, n_input, n_output, hidden_sizes):
        super(Net_ReLU, self).__init__()
        layers = [n_input] + hidden_sizes + [n_output]
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
    def forward(self, x):
        for i, l in enumerate(self.linears):
            if i < len(self.linears) - 1:
                x = F.relu(l(x))
            else:
                x = l(x)
        return x


# ---- Data Loading ----
with open(conf["gel_file"], "r") as f:
    gel_data = json.load(f)
with open(conf["opt_file"], "r") as f:
    rel2_data = json.load(f)
with open(conf["ram_file"], "r") as f:
    rel1_data = json.load(f)

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
IN1 = 50  # Set a fixed value for testing
IN2 = rel2_data["IN"]
IN2 = 50  # Set a fixed value for testing
IN = IN1 + IN2

# ---- Tensor Creation ----
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

Pw_h = torch.cat((Pw1, Pw2), dim=0)
ld_h = torch.cat((ld1, ld2), dim=0)

opt_u = torch.cat((opt_u2, opt_u1), dim=0)
Pw = torch.cat((Pw2, Pw1), dim=0)
ld = torch.cat((ld2, ld1), dim=0)
x_tmp = torch.cat((opt_u.view(IN, G*T), Pw.view(IN, W*T)), dim=1)
x_tmp = torch.cat((x_tmp, ld.view(IN, D*T)), dim=1)
x = Variable(x_tmp)

result_y = []

for yy in conf["test_idxs"]:
    u_test = np.zeros((G, T))
    Pw_test = torch.tensor(rel2_data[f"Pw{yy}"]).view(W, T)
    for i in range(G+1):
        for j in range(T+1):
            u_test[i-1, j-1] = rel2_data[f"opt_u{conf['pw_test_idx']}"][j-1][i-1]
    y_test = torch.tensor(rel2_data[f"scd_obj{conf['scd_test_idx']}"])
    print(rel2_data.keys())
    ld_test = torch.tensor(rel2_data[f"load{conf['ld_test_idx']}"]).view(D, T)

    fstf_n = G*T
    scdf_n = W*T
    ld_n = D*T
    n_output = 1
    n_input = W*T + G*T + D*T

    Pw_mx = Pw_h.view(IN, W*T)
    ld_mx = ld_h.view(IN, D*T)

    # --- Load trained network ---
    my_nn = Net_ReLU(n_input, n_output, conf["arch"])
    my_nn.load_state_dict(torch.load(conf["nn_params"]))

    input_bounds = {}
    for i in range(fstf_n):
        input_bounds[i] = (0, 1)
    for i in range(scdf_n):
        input_bounds[fstf_n + i] = (0, Pw_mx[:, i].max().item())
    for i in range(ld_n):
        input_bounds[fstf_n + scdf_n + i] = (0, ld_mx[:, i].max().item())

    x_t = x[1].view(1, -1)
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            my_nn,
            x_t,
            f,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        write_onnx_model_with_bounds(f.name, None, input_bounds)
        print(f"Wrote PyTorch model to {f.name}")
        network_definition = load_onnx_neural_network_with_bounds(f.name)

    # ---- Build Pyomo Model ----
    model_full = pyo.ConcreteModel()

    # Binary variables
    model_full.s = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.Binary, bounds=(0, 1), initialize=0)
    model_full.d = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.Binary, bounds=(0, 1), initialize=0)
    model_full.u = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.Binary, bounds=(0, 1))

    # Continuous variables
    model_full.y = pyo.Var(initialize=0)
    model_full.Chs = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.NonNegativeReals)
    model_full.Ces = pyo.Var(pyo.RangeSet(G), pyo.RangeSet(T), domain=pyo.NonNegativeReals)
    model_full.Pw = pyo.Var(pyo.RangeSet(W), pyo.RangeSet(T), domain=pyo.NonNegativeReals)
    model_full.ld = pyo.Var(pyo.RangeSet(D), pyo.RangeSet(T), domain=pyo.NonNegativeReals)

    model_full.nn = OmltBlock()
    formulation_full = ReluBigMFormulation(network_definition)
    model_full.nn.build_formulation(formulation_full)

    # Constraints (unchanged from your original code; you could refactor further for DRYness)
    def constraint_hs_rule(mdl, i, t):
        return mdl.Chs[i, t] == Chot[i-1] * mdl.s[i, t]

    def constraint_es_rule(mdl, i, t, taoi, fit):
        return mdl.Ces[i, t] >= (Ccold[i-1] - Chot[i-1]) * (mdl.s[i, t] - sum(mdl.d[i, k] for k in range(taoi, t)) - fit)

    def constraint_initial_state_rule(mdl, i, t):
        return mdl.u[i, t] == x0[i-1]

    def constraint_start_state_rule(mdl, i, t, omiga_on):
        return sum(mdl.s[i, k] for k in range(omiga_on, t+1)) <= mdl.u[i, t]

    def constraint_shut_down_state_rule(mdl, i, t, omiga_off):
        return sum(mdl.d[i, k] for k in range(omiga_off, t+1)) <= 1 - mdl.u[i, t]

    def constraint_units_state_balance_rule(mdl, i, t):
        if t == 1:
            return mdl.s[i, t] - mdl.d[i, t] == mdl.u[i, t] - x0[i-1]
        else:
            return mdl.s[i, t] - mdl.d[i, t] == mdl.u[i, t] - mdl.u[i, t-1]
        
    def constraint_x_rule(mdl, i, t):
        return mdl.u[i, t] == mdl.nn.inputs[(i-1)*T + t - 1]

    def constraint_y_rule(mdl):
        return mdl.y == mdl.nn.outputs[0]

    def constraint_Pw_rule(mdl, w, t):
        return mdl.Pw[w, t] == mdl.nn.inputs[G*T + (w-1)*T + t - 1]

    def constraint_PwSce_rule(mdl, w, t):
        return mdl.Pw[w, t] == Pw_test[w-1, t-1].item()

    def constraint_ld_rule(mdl, d, t):
        return mdl.ld[d, t] == mdl.nn.inputs[G*T + W*T + (d-1)*T + t - 1]

    def constraint_ldSce_rule(mdl, d, t):
        return mdl.ld[d, t] == ld_test[d-1, t-1].item()

    for i in range(1, G+1):
        for t in range(1, T+1):
            taoi = int(max(1, t-Toff[i-1]-Tcold[i-1]))
            fit = 1 if (t-taoi <= 0) and (max(0, -T0[i-1]) < abs(t-taoi) + 1) else 0
            omiga_on = max(0, t-Ton[i-1]) + 1
            omiga_off = max(0, t-Toff[i-1]) + 1
            model_full.add_component(f'constraint_hs_{i}_{t}', pyo.Constraint(rule=constraint_hs_rule(model_full, i, t)))
            model_full.add_component(f'constraint_es_{i}_{t}', pyo.Constraint(rule=constraint_es_rule(model_full, i, t, taoi, fit)))
            if t >= 1 and t <= U0[i-1]+L0[i-1]:
                model_full.add_component(f'constraint_initial_state_{i}_{t}', pyo.Constraint(rule=constraint_initial_state_rule(model_full, i, t)))
            model_full.add_component(f'constraint_start_state_{i}_{t}', pyo.Constraint(rule=constraint_start_state_rule(model_full, i, t, omiga_on)))
            if omiga_off <= t and t >= 1+L0[i-1]:
                model_full.add_component(f'constraint_shut_down_state_{i}_{t}', pyo.Constraint(rule=constraint_shut_down_state_rule(model_full, i, t, omiga_off)))
            model_full.add_component(f'constraint_units_state_balance_{i}_{t}', pyo.Constraint(rule=constraint_units_state_balance_rule(model_full, i, t)))
            model_full.add_component(f'connect_inputs_{i}_{t}', pyo.Constraint(rule=constraint_x_rule(model_full, i, t)))

    for w in range(1, W+1):
        for t in range(1, T+1):
            model_full.add_component(f'constraint_Pw_{w}_{t}', pyo.Constraint(rule=constraint_Pw_rule(model_full, w, t)))
            model_full.add_component(f'constraint_PwSce_{w}_{t}', pyo.Constraint(rule=constraint_PwSce_rule(model_full, w, t)))
    for d in range(1, D+1):
        for t in range(1, T+1):
            model_full.add_component(f'constraint_ld_{d}_{t}', pyo.Constraint(rule=constraint_ld_rule(model_full, d, t)))
            model_full.add_component(f'constraint_ldSce_{d}_{t}', pyo.Constraint(rule=constraint_ldSce_rule(model_full, d, t)))
    model_full.add_component(f'connect_outputs', pyo.Constraint(rule=constraint_y_rule(model_full)))

    model_full.obj = pyo.Objective(expr=(model_full.y) + sum(model_full.Chs[:, :]) + sum(model_full.Ces[:, :]), sense=pyo.minimize)

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
    y_results = pyo.value(model_full.y)
    fst_obj = sum(pyo.value(model_full.Chs[:,:])) + sum(pyo.value(model_full.Ces[:,:]))

    print("u_nn = ", u_rlt)
    print("u_test = ", u_test)
    print("y_nn = ", y_results)
    print("y_test = ", y_test)
    result_y.append(y_results)

print(result_y)
