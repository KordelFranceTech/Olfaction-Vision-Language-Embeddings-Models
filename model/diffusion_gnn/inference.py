import torch


from .diffusion_gnn import sample, validate_molecule
from .diffusion_gnn import load_goodscents_subset, smiles_to_graph
from .diffusion_gnn import EGNNDiffusionModel, OlfactoryConditioner


def test_models(test_model, test_conditioner):
    good_count: int = 0
    index: int = 10
    smiles_list, label_map, label_names = load_goodscents_subset(index=index)
    num_labels = len(label_names)
    dataset = []
    model.eval()
    conditioner.eval()
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g:
            g.y = torch.tensor(label_map[smi])
            dataset.append(g)

    for i in range(0, len(dataset)):
        print(f"Testing molecule {i+1}/{len(dataset)}")
        data = dataset[i]
        x_0, pos, edge_index, edge_attr, label_vec = data.x, data.pos, data.edge_index, data.edge_attr.view(-1), data.y
        print(f"label vec: {label_vec}")
        print(f"len: {len(label_vec.tolist())}")
        new_smiles = sample(test_model, test_conditioner, label_vec=label_vec)
        print(new_smiles)
        valid, props = validate_molecule(new_smiles)
        print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")
        if new_smiles != "":
            good_count += 1
    percent_correct: float = float(good_count)  / float(len(dataset))
    print(f"Percent correct: {percent_correct}")


def inference(model: EGNNDiffusionModel, conditioner: OlfactoryConditioner, data):
    model.eval()
    conditioner.eval()
    x_0, pos, edge_index, edge_attr, label_vec = data.x, data.pos, data.edge_index, data.edge_attr.view(-1), data.y
    new_smiles = sample(model, conditioner, label_vec=label_vec)
    print(new_smiles)
    valid, props = validate_molecule(new_smiles)
    print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")


def load_models():
    smiles_list, label_map, label_names = load_goodscents_subset(index=0)
    num_labels = len(label_names)
    model = EGNNDiffusionModel(node_dim=1, embed_dim=8)
    model.load_state_dict(torch.load('/content/egnn_state_dict_20250427.pth'))
    model.eval() # Set to evaluation mode if you are not training

    conditioner = OlfactoryConditioner(num_labels=num_labels, embed_dim=8)
    conditioner.load_state_dict(torch.load('/content/olfactory_conditioner_state_dict.pth'))
    conditioner.eval() # Set to evaluation mode if you are not training

    return model, conditioner


if __name__ == "__main__":
    model, conditioner = load_models()
    test_models(test_model=model, test_conditioner=conditioner)
