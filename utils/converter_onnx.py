import glob, torch
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
from torch.utils.data import DataLoader

fname = 'model.onnx'
model_path='trainings/particlenet_pf/20221216-120722_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_100_epoch_15_cat012/net_best_epoch_state.pt'
dev = torch.device('cpu')

path_network_config = 'models/particlenet_pf.py'
path_data_config = 'data/VBF_points_features.yaml'
network_module = import_module(path_network_config, name='_network_module')

data_config = SimpleIterDataset({}, path_data_config, for_training=False).config
model, model_info = network_module.get_model(data_config,for_inference = True)

model_state = torch.load(model_path, map_location='cpu')
model.load_state_dict(model_state)
model.eval()

print(model_info)
inputs = tuple(torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
opset_version=12
torch.onnx.export(model, inputs, model_path.replace('net_best_epoch_state.pt',fname.replace('.onnx','_ops'+str(opset_version)+'.onnx')), input_names=model_info['input_names'], output_names=model_info['output_names'], dynamic_axes=model_info.get('dynamic_axes', None), opset_version=opset_version)
