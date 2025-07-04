import json, sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils
import wandb  # optional, comment out if not used

# ---- Config ----
CASE_CONFIGS = {
    "A": {
        "gel_file": "data/caseA_gel.json",
        "opt_file": "results/opt_case_A.json",
        "ram_file": "results/ram_case_A.json",
        "nn_params": 'models/net_params_caseA.pkl',
        "arch": [100, 60, 30, 15, 4],
        "wandb_project": "tssnetcaseA_multi",
        "sweep_epochs": 500,
        "sweep_batch_min": 10,
        "sweep_batch_max": 20,
        "sweep_count": 3,
    },
    "B": {
        "gel_file": "data/caseB_gel.json",
        "opt_file": "results/opt_case_B.json",
        "ram_file": "results/ram_case_B.json",
        "nn_params": 'models/net_params_caseB.pkl',
        "arch": [180, 120, 60, 30, 15, 4],
        "wandb_project": "tssnetcaseB_multi",
        "sweep_epochs": 6000,
        "sweep_batch_min": 20,
        "sweep_batch_max": 620,
        "sweep_count": 6,
    },
    "C": {
        "gel_file": "data/caseC_gel.json",
        "opt_file": "results/opt_case_C.json",
        "ram_file": "results/ram_case_C.json",
        "nn_params": 'my_omlt/networks/net_params_caseC.pkl',
        "arch": [360, 180, 90, 45, 20, 10],
        "wandb_project": "tssnetcaseC_multi",
        "sweep_epochs": 5000,
        "sweep_batch_min": 100,
        "sweep_batch_max": 2000,
        "sweep_count": 6,
    }
}

case = sys.argv[1] if len(sys.argv) > 1 else "B"
conf = CASE_CONFIGS[case]

# ---- NN definition (configurable layers) ----
class Net_ReLU(torch.nn.Module):
    def __init__(self, n_input, n_output, arch):
        super(Net_ReLU, self).__init__()
        layers = [n_input] + arch + [n_output]
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
    def forward(self, x):
        for l in self.linears[:-1]:
            x = F.relu(l(x))
        x = self.linears[-1](x)
        return x

# ---- RMSE loss (normalized) ----
class RMSEloss(torch.nn.Module):
    def __init__(self, y_max, y_min):
        super(RMSEloss, self).__init__()
        self.y_max = y_max
        self.y_min = y_min
    def forward(self, x, y):
        rmse = torch.sqrt(torch.mean((x - y) ** 2))
        max_possible = torch.sqrt(torch.tensor((self.y_max - self.y_min) ** 2))
        return rmse / max_possible

# ---- Data loading ----
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

IN1 = rel1_data["IN"]
IN1 = 50
IN2 = rel2_data["IN"]
IN2 = 50
IN = IN1 + IN2


def load_tensor_group(data, IN, G, T, W, D):
    opt_u = torch.zeros(IN, G*T, 1)
    ld    = torch.zeros(IN, D*T, 1)
    Pw    = torch.zeros(IN, W*T, 1)
    scd_obj = torch.zeros(IN, 1)
    for it in range(IN):
        opt_u[it, :, :] = torch.tensor(data[f"opt_u{it+1}"]).view(-1, 1)
        ld[it, :, :] = torch.tensor(data[f"load{it+1}"]).view(-1, 1)
        Pw[it, :, :] = torch.tensor(data[f"Pw{it+1}"]).view(-1, 1)
        scd_obj[it, :] = torch.tensor(data[f"scd_obj{it+1}"]).unsqueeze(0)
    return opt_u, ld, Pw, scd_obj

opt_u1, ld_1, Pw1, scd_obj1 = load_tensor_group(rel1_data, IN1, G, T, W, D)
opt_u2, ld_2, Pw2, scd_obj2 = load_tensor_group(rel2_data, IN2, G, T, W, D)

n_input = W*T + G*T + D*T
n_output = 1

opt_u = torch.cat((opt_u1, opt_u2), dim=0)
scd_obj = torch.cat((scd_obj1, scd_obj2), dim=0)
Pw = torch.cat((Pw1, Pw2), dim=0)
ld = torch.cat((ld_1, ld_2), dim=0)
x_tmp = torch.cat((opt_u.view(IN, G*T), Pw.view(IN, W*T)), dim=1)
x_tmp = torch.cat((x_tmp, ld.view(IN, D*T)), dim=1)
x, y = Variable(x_tmp), Variable(scd_obj)

y_max = y.max().item()
y_min = y.min().item()

combined = torch.utils.data.TensorDataset(x, y)
data_loader = torch.utils.data.DataLoader(combined, batch_size=IN, shuffle=True)
for shuffled_data in data_loader:
    x_f, y_f = shuffled_data
x_f = torch.cat((x[0,:].unsqueeze(0), x_f), dim=0)
y_f = torch.cat((y[0,:].unsqueeze(0), y_f), dim=0)
T_N = int(IN*0.7)
dataset1 = torch.utils.data.TensorDataset(x_f[0:T_N,:], y_f[0:T_N,:])
dataset2 = torch.utils.data.TensorDataset(x_f[T_N:IN,:], y_f[T_N:IN,:])

# ---- Sweep configuration for wandb ----
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'max': 0.001, 'min': 0.00001, 'distribution': 'uniform'},
        'batch_size': {'max': conf["sweep_batch_max"], 'min': conf["sweep_batch_min"], 'distribution': 'int_uniform'},
    },
}
# ---- Training function for wandb sweep ----
def train(dataset1, dataset2, y_max, y_min):
    wandb.init()
    config = wandb.config
    model = Net_ReLU(n_input, n_output, conf["arch"])
    criterion = RMSEloss(y_max, y_min)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset2, batch_size=config.batch_size, shuffle=True)
    best_val_loss = float('inf')
    for epoch in range(conf["sweep_epochs"]):
        model.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                wandb.log({'training_loss': loss.item()})
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, target in val_loader:
            output = model(x)
            val_loss += criterion(output, target).item()
    val_loss /= len(val_loader)
    wandb.log({'val_loss': val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), conf["nn_params"])
        #wandb.save(conf["nn_params"])
        print("current best validation loss: ", best_val_loss)

# ---- Run sweep ----
sweep_id = wandb.sweep(sweep_config, project=conf["wandb_project"])
wandb.agent(sweep_id, function=lambda: train(dataset1, dataset2, y_max, y_min), count=conf["sweep_count"])

# --- End of nn_train.py ---
