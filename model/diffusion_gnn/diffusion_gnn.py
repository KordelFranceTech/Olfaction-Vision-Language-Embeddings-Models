import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import math

# -------- UTILS: Molecule Processing with 3D Coordinates --------
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None

    conf = mol.GetConformer()
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    node_feats = []
    pos = []
    edge_index = []
    edge_attrs = []

    for atom in atoms:
        # Normalize atomic number
        node_feats.append([atom.GetAtomicNum() / 100.0])
        position = conf.GetAtomPosition(atom.GetIdx())
        pos.append([position.x, position.y, position.z])

    for bond in bonds:
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
        bond_type = bond.GetBondType()
        bond_class = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }.get(bond_type, 0)
        edge_attrs.extend([[bond_class], [bond_class]])

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        pos=torch.tensor(pos, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.long)
    )

# -------- EGNN Layer --------
class EGNNLayer(MessagePassing):
    def __init__(self, node_dim):
        super().__init__(aggr='add')
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, node_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, pos, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self.coord_updates = torch.zeros_like(pos)
        x_out, coord_out = self.propagate(edge_index, x=x, pos=pos)
        return x_out, pos + coord_out

    def message(self, x_i, x_j, pos_i, pos_j):
        edge_vec = pos_j - pos_i
        dist = ((edge_vec**2).sum(dim=-1, keepdim=True) + 1e-8).sqrt()
        h = torch.cat([x_i, x_j, dist], dim=-1)
        edge_msg = self.node_mlp(h)
        coord_update = self.coord_mlp(dist) * edge_vec
        return edge_msg, coord_update

    def message_and_aggregate(self, adj_t, x):
        raise NotImplementedError("This EGNN layer does not support sparse adjacency matrices.")

    def aggregate(self, inputs, index):
        edge_msg, coord_update = inputs
        aggr_msg = torch.zeros(index.max() + 1, edge_msg.size(-1), device=edge_msg.device).index_add_(0, index, edge_msg)
        aggr_coord = torch.zeros(index.max() + 1, coord_update.size(-1), device=coord_update.device).index_add_(0, index, coord_update)
        return aggr_msg, aggr_coord

    def update(self, aggr_out, x):
        msg, coord_update = aggr_out
        return x + msg, coord_update

# -------- Time Embedding --------
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, t):
        return self.net(t.view(-1, 1).float() / 1000)

# -------- Olfactory Conditioning --------
class OlfactoryConditioner(nn.Module):
    def __init__(self, num_labels, embed_dim):
        super().__init__()
        self.embedding = nn.Linear(num_labels, embed_dim)

    def forward(self, labels):
        return self.embedding(labels.float())

# -------- EGNN Diffusion Model --------
class EGNNDiffusionModel(nn.Module):
    def __init__(self, node_dim, embed_dim):
        super().__init__()
        self.time_embed = TimeEmbedding(embed_dim)
        self.egnn1 = EGNNLayer(node_dim + embed_dim * 2)
        self.egnn2 = EGNNLayer(node_dim + embed_dim * 2)
        self.bond_predictor = nn.Sequential(
            nn.Linear((node_dim + embed_dim * 2) * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x_t, pos, edge_index, t, cond_embed):
        batch_size = x_t.size(0)
        t_embed = self.time_embed(t).expand(batch_size, -1)
        cond_embed = cond_embed.expand(batch_size, -1)
        x_input = torch.cat([x_t, cond_embed, t_embed], dim=1)
        x1, pos1 = self.egnn1(x_input, pos, edge_index)
        x2, pos2 = self.egnn2(x1, pos1, edge_index)
        edge_feats = torch.cat([x2[edge_index[0]], x2[edge_index[1]]], dim=1)
        bond_logits = self.bond_predictor(edge_feats)
        return x2[:, :x_t.shape[1]], bond_logits

# -------- Noise and Training --------
def add_noise(x_0, noise, t):
    return x_0 + noise * (t / 1000.0)


def plot_data(mu, sigma, color, title):
    all_losses = np.array(mu)
    sigma_losses = np.array(sigma)
    x = np.arange(len(mu))
    plt.plot(x, all_losses, f'{color}-')
    plt.fill_between(x, all_losses - sigma_losses, all_losses + sigma_losses, color=color, alpha=0.2)
    plt.legend(['Mean Loss', 'Variance of Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()


def train(model, conditioner, dataset, epochs=10):
    model.train()
    conditioner.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(conditioner.parameters()), lr=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    all_bond_losses: list = []
    all_noise_losses: list = []
    all_losses: list = []
    all_sigma_bond_losses: list = []
    all_sigma_noise_losses: list = []
    all_sigma_losses: list = []

    for epoch in range(epochs):
        total_bond_loss = 0
        total_noise_loss = 0
        total_loss = 0
        sigma_bond_losses: list = []
        sigma_noise_losses: list = []
        sigma_losses: list = []

        for data in dataset:
            x_0, pos, edge_index, edge_attr, labels = data.x, data.pos, data.edge_index, data.edge_attr.view(-1), data.y
            if torch.any(edge_attr >= 4) or torch.any(edge_attr < 0) or torch.any(torch.isnan(x_0)):
                continue  # skip corrupted data
            # if torch.any(edge_attr < 0) or torch.any(torch.isnan(x_0)):
            #     continue  # skip corrupted data
            # print(f"x0: {x_0}")
            t = torch.tensor([random.randint(1, 1000)])
            noise = torch.randn_like(x_0) # original
            # noise = torch.rand_like(x_0) # mine
            x_t = add_noise(x_0, noise, t)
            # x_t.relu_()
            # print(f"\tx_t: {x_t}")
            cond_embed = conditioner(labels)
            # print(f"\tcond_embed: {cond_embed}")
            pred_noise, bond_logits = model(x_t, pos, edge_index, t, cond_embed)
            # print(f"\tpred_noise: {pred_noise}\n\tbond logits: {bond_logits}")
            # Suppress this if needed. This is optimization, not necessity
            # bond_logits = temperature_scaled_softmax(bond_logits, temperature=(1 - (1/(epoch+1))))
            # loss = F.mse_loss(pred_noise, noise) + ce_loss(bond_logits, edge_attr)
            loss_noise = F.mse_loss(pred_noise, noise)
            loss_bond = ce_loss(bond_logits, edge_attr)
            loss = loss_noise + loss_bond
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_bond_loss += loss_bond.item()
            total_noise_loss += loss_noise.item()
            total_loss += loss.item()
            sigma_bond_losses.append(loss_bond.item())
            sigma_noise_losses.append(loss_noise.item())
            sigma_losses.append(loss.item())

        all_bond_losses.append(total_bond_loss)
        all_noise_losses.append(total_noise_loss)
        all_losses.append(total_loss)
        all_sigma_bond_losses.append(torch.std(torch.tensor(sigma_bond_losses)))
        all_sigma_noise_losses.append(torch.std(torch.tensor(sigma_noise_losses)))
        all_sigma_losses.append(torch.std(torch.tensor(sigma_losses)))
        # print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Noise Loss = {total_noise_loss:.4f}, Bond Loss = {total_bond_loss:.4f}")

    plot_data(mu=all_bond_losses, sigma=all_sigma_bond_losses, color='b', title="Bond Loss")
    plot_data(mu=all_noise_losses, sigma=all_sigma_noise_losses, color='r', title="Noise Loss")
    plot_data(mu=all_losses, sigma=all_sigma_losses, color='g', title="Total Loss")

    plt.plot(all_bond_losses)
    plt.plot(all_noise_losses)
    plt.plot(all_losses)
    plt.legend(['Bond Loss', 'Noise Loss', 'Total Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()
    return model, conditioner


# -------- Generation --------
def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)


from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

def sample_batch(model, conditioner, label_vec, steps=1000, batch_size=4):
    mols = []
    for _ in range(batch_size):
        x_t = torch.randn((10, 1))
        pos = torch.randn((10, 3))
        edge_index = torch.randint(0, 10, (2, 20))

        for t in reversed(range(1, steps + 1)):
            cond_embed = conditioner(label_vec.unsqueeze(0))
            pred_x, bond_logits = model(x_t, pos, edge_index, torch.tensor([t]), cond_embed)
            x_t = x_t - pred_x * (1.0 / steps)

        x_t = x_t * 100.0
        x_t.relu_()
        atom_types = torch.clamp(x_t.round(), 1, 118).int().squeeze().tolist()
        allowed_atoms = [6, 7, 8, 9, 15, 16, 17]  # C, N, O, F, P, S, Cl
        bond_logits.relu_()

        mol = Chem.RWMol()
        idx_map = {}
        for i, atomic_num in enumerate(atom_types):
            if atomic_num not in allowed_atoms:
                continue
            try:
                atom = Chem.Atom(int(atomic_num))
                idx_map[i] = mol.AddAtom(atom)
            except Exception:
                continue

        if len(idx_map) < 2:
            continue

        bond_type_map = {
            0: Chem.BondType.SINGLE,
            1: Chem.BondType.DOUBLE,
            2: Chem.BondType.TRIPLE,
            3: Chem.BondType.AROMATIC
        }

        added = set()
        for i in range(edge_index.shape[1]):
            a = int(edge_index[0, i])
            b = int(edge_index[1, i])
            if a != b and (a, b) not in added and (b, a) not in added and a in idx_map and b in idx_map:
                try:
                    bond_type = bond_type_map.get(bond_preds[i], Chem.BondType.SINGLE)
                    mol.AddBond(idx_map[a], idx_map[b], bond_type)
                    added.add((a, b))
                except Exception:
                    continue

        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mols.append(mol)
        except Exception:
            continue

    # if mols:
    #     img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), legends=[Chem.MolToSmiles(m) for m in mols])
    #     img.save("generated_image.png")
    #     img.show()
    # else:
    #     print("No valid molecules were generated.")
    return mols


def sample(model, conditioner, label_vec, steps=1000, debug=True):
    x_t = torch.randn((10, 1))
    pos = torch.randn((10, 3))
    edge_index = torch.randint(0, 10, (2, 20))

    for t in reversed(range(1, steps + 1)):
        cond_embed = conditioner(label_vec.unsqueeze(0))
        pred_x, bond_logits = model(x_t, pos, edge_index, torch.tensor([t]), cond_embed)
        bond_logits = temperature_scaled_softmax(bond_logits, temperature=(1/t))
        x_t = x_t - pred_x * (1.0 / steps)

    x_t = x_t * 100.0
    x_t.relu_()
    atom_types = torch.clamp(x_t.round(), 1, 118).int().squeeze().tolist()
    ## Try limiting to only the molecules that the Scentience sensors can detect
    allowed_atoms = [6, 7, 8, 9, 15, 16, 17]  # C, N, O, F, P, S, Cl
    bond_logits.relu_()
    bond_preds = torch.argmax(bond_logits, dim=-1).tolist()
    # bond_preds = torch.tensor(bond_preds)
    # bond_preds = bond_preds * 100.0
    # bond_preds.relu_()
    # bond_preds.abs_()
    # bond_preds = bond_preds.round().int().tolist()
    if debug:
        print(f"\tcond_embed: {cond_embed}")
        print(f"\tx_t: {x_t}")
        print(f"\tprediction: {x_t}")
        print(f"\tbond logits: {bond_logits}")
        print(f"\tatoms: {atom_types}")
        print(f"\tbonds: {bond_preds}")

    mol = Chem.RWMol()
    idx_map = {}
    for i, atomic_num in enumerate(atom_types):
        if atomic_num not in allowed_atoms:
            continue
        try:
            atom = Chem.Atom(int(atomic_num))
            idx_map[i] = mol.AddAtom(atom)
        except Exception:
            continue

    if len(idx_map) < 2:
        print("Molecule too small or no valid atoms after filtering.")
        return ""

    bond_type_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC
    }

    added = set()
    for i in range(edge_index.shape[1]):
        a = int(edge_index[0, i])
        b = int(edge_index[1, i])
        if a != b and (a, b) not in added and (b, a) not in added and a in idx_map and b in idx_map:
            try:
                bond_type = bond_type_map.get(bond_preds[i], Chem.BondType.SINGLE)
                mol.AddBond(idx_map[a], idx_map[b], bond_type)
                added.add((a, b))
            except Exception:
                continue
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        img = Draw.MolToImage(mol)
        img.show()
        print(f"Atom types: {atom_types}")
        print(f"Generated SMILES: {smiles}")
        return smiles
    except Exception as e:
        print(f"Sanitization error: {e}")
        return ""

"""
Same as sample_a but with fixes:
    * Hydrogen atoms (and other unstable elements) are now excluded from generation.
    * Molecules with fewer than two valid atoms are skipped early.
    * RDKit warnings (like valence errors) are suppressed from terminal output.
Note:
    * You're filtering for atoms in [6, 7, 8, 9, 15, 16, 17] (C, N, O, F, P, S, Cl) — which is reasonable — but:
        Some molecule graphs may become disconnected or too small (<2 atoms) after filtering
        You're skipping all such graphs instead of attempting to repair or relax them
"""
def sample_original(model, conditioner, label_vec, steps=1000):
    x_t = torch.randn((10, 1))
    pos = torch.randn((10, 3))
    edge_index = torch.randint(0, 10, (2, 20))

    for t in reversed(range(1, steps + 1)):
        cond_embed = conditioner(label_vec.unsqueeze(0))
        pred_x, bond_logits = model(x_t, pos, edge_index, torch.tensor([t]), cond_embed)
        # bond_logits = temperature_scaled_softmax(bond_logits, temperature=(1/t))
        x_t = x_t - pred_x * (1.0 / steps)

    atom_types = torch.clamp(x_t.round(), 1, 118).int().squeeze().tolist()
    allowed_atoms = [6, 7, 8, 9, 15, 16, 17]  # C, N, O, F, P, S, Cl
    bond_preds = torch.argmax(bond_logits, dim=-1).tolist()

    mol = Chem.RWMol()
    idx_map = {}
    for i, atomic_num in enumerate(atom_types):
        if atomic_num not in allowed_atoms:
            continue
        try:
            atom = Chem.Atom(int(atomic_num))
            idx_map[i] = mol.AddAtom(atom)
        except Exception:
            continue

    if len(idx_map) < 2:
        print("Molecule too small or no valid atoms after filtering.")
        return ""

    bond_type_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC
    }

    added = set()
    for i in range(edge_index.shape[1]):
        a = int(edge_index[0, i])
        b = int(edge_index[1, i])
        if a != b and (a, b) not in added and (b, a) not in added and a in idx_map and b in idx_map:
            try:
                bond_type = bond_type_map.get(bond_preds[i], Chem.BondType.SINGLE)
                mol.AddBond(idx_map[a], idx_map[b], bond_type)
                added.add((a, b))
            except Exception:
                continue
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        img = Draw.MolToImage(mol)
        img.show()
        print(f"Atom types: {atom_types}")
        print(f"Generated SMILES: {smiles}")
        return smiles
    except Exception as e:
        print(f"Sanitization error: {e}")
        return ""


def sample_a(model, conditioner, label_vec, steps=500):
    from rdkit.Chem import Draw
    x_t = torch.randn((10, 1))
    pos = torch.randn((10, 3))
    edge_index = torch.randint(0, 10, (2, 20))
    model.eval()

    for t in reversed(range(1, steps + 1)):
        cond_embed = conditioner(label_vec.unsqueeze(0))
        pred_x, bond_logits = model(x_t, pos, edge_index, torch.tensor([t]), cond_embed)
        x_t = x_t - pred_x * (1.0 / steps)

    atom_types = torch.clamp(x_t.round() * 100, 1, 118).int().squeeze().tolist()
    bond_preds = torch.argmax(bond_logits, dim=-1).tolist()

    mol = Chem.RWMol()
    idx_map = {}
    for i, atomic_num in enumerate(atom_types):
        try:
            atom = Chem.Atom(int(atomic_num))
            idx_map[i] = mol.AddAtom(atom)
        except Exception:
            continue

    bond_type_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC
    }

    added = set()
    for i in range(edge_index.shape[1]):
        a = int(edge_index[0, i])
        b = int(edge_index[1, i])
        if a != b and (a, b) not in added and (b, a) not in added and a in idx_map and b in idx_map:
            try:
                bond_type = bond_type_map.get(bond_preds[i], Chem.BondType.SINGLE)
                mol.AddBond(idx_map[a], idx_map[b], bond_type)
                added.add((a, b))
            except Exception:
                continue

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        img = Draw.MolToImage(mol)
        img.show()
        return smiles
    except:
        return ""

# -------- Validation --------
def validate_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, {}
    return True, {"MolWt": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol)}

# -------- Load Data --------
def load_goodscents_subset(filepath="/content/curated_GS_LF_merged_4983.csv",
                           index=200
                           ):
    # max_rows = 500
    df = pd.read_csv(filepath)
    # df = df.sample(frac=1).reset_index(drop=True)
    if index > 0:
        df = df.head(index)
    else:
        df = df.tail(-1*index)
    descriptor_cols = df.columns[2:]
    smiles_list, label_map = [], {}
    for _, row in df.iterrows():
        smiles = row["nonStereoSMILES"]
        labels = row[descriptor_cols].astype(int).tolist()
        if smiles and any(labels):
            smiles_list.append(smiles)
            label_map[smiles] = labels
        # if len(smiles_list) >= index:
        #     break
    return smiles_list, label_map, list(descriptor_cols)


# -------- Main --------
if __name__ == '__main__':
    SHOULD_BATCH: bool = False
    smiles_list, label_map, label_names = load_goodscents_subset(index=500)
    num_labels = len(label_names)
    dataset = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g:
            g.y = torch.tensor(label_map[smi])
            dataset.append(g)
    model = EGNNDiffusionModel(node_dim=1, embed_dim=8)
    conditioner = OlfactoryConditioner(num_labels=num_labels, embed_dim=8)
    # train(model, conditioner, dataset, epochs=500)
    train_success: bool = False
    while not train_success:
        try:
            model, conditioner = train(model, conditioner, dataset, epochs=1000)
            train_success = True
            break
        except IndexError:
            print("Index Error on training. Trying again.")
    test_label_vec = torch.zeros(num_labels)
    if "floral" in label_names:
        test_label_vec[label_names.index("floral")] = 0
    if "fruity" in label_names:
        test_label_vec[label_names.index("fruity")] = 1
    if "musky" in label_names:
        test_label_vec[label_names.index("musky")] = 0

    model.eval()
    conditioner.eval()
    if SHOULD_BATCH:
        new_smiles_list = sample_batch(model, conditioner, label_vec=test_label_vec)
        for new_smiles in new_smiles_list:
            print(new_smiles)
            valid, props = validate_molecule(new_smiles)
            print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")
    else:
        new_smiles = sample(model, conditioner, label_vec=test_label_vec)
        print(new_smiles)
        valid, props = validate_molecule(new_smiles)
        print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")
