import glob, torch
import onnxruntime as ort
import numpy as np
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
from torch.utils.data import DataLoader
model_path='trainings/particlenet_pf/20221111-181843_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_epoch_40_cat012/net_best_epoch_state.pt'
model_path='trainings/particlenet_pf/20221216-120722_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_100_epoch_15_cat012/net_best_epoch_state.pt'
model_onnx=glob.glob(model_path.replace('net_best_epoch_state.pt','*onnx'))
if len(model_onnx)>1:
  raise RuntimeError("More that 1 onnx model found,"+str(model_onnx))
else:
  model_onnx= model_onnx[0]
dev = torch.device('cpu')

path_network_config = 'models/particlenet_pf.py'
path_data_config = 'data/VBF_points_features.yaml'
network_module = import_module(path_network_config, name='_network_module')

data_config = SimpleIterDataset({}, path_data_config, for_training=False).config
model, model_info = network_module.get_model(data_config)
model.eval()
model_state = torch.load(model_path, map_location='cpu')
model.load_state_dict(model_state)

filelist = glob.glob('/eos/home-a/anmalara/Public/DNNInputs/eventCategory_0/MC__*_standard_UL1*.root')
filelist = filelist[0:1]
num_workers = min(4,len(filelist))
batch_size = 1

test_data = SimpleIterDataset({'prediction': filelist}, path_data_config, for_training=False, fetch_by_files=True, fetch_step=1, name='prediction')
test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, drop_last=False, pin_memory=True)

ort_sess = ort.InferenceSession(model_onnx)

for inputs_, labels_, observers_ in test_loader:
  print("next")
  label = labels_[data_config.label_names[0]].long()
  inputs_n = {k: v.cpu().numpy() for k, v in inputs_.items()}
  inputs = [inputs_[k].to(dev) for k in data_config.input_names]
  model_output = model(*inputs)
  model_output = torch.softmax(model_output, dim=1).detach().cpu().numpy()
  score = ort_sess.run([], inputs_n)[0]
  print('output -> onnx:', score, ' --> model:', model_output, ' diff-> ', score[0]-model_output[0])


# Additional test


eta = [-2.8023, -2.8151,  2.3601,  2.7928, -3.0821,  2.3709, -3.0821, -2.8097,
-3.2350, -3.0622,  2.6190,  2.3792,  2.8042, -2.9175, -2.8139,  4.3079,
4.4831, -3.2588,  2.8366,  2.8778, -1.9842,  1.3261,  2.8214,  2.7322,
2.8728, -4.1303, -2.5764, -2.7595, -0.4827,  0.2807, -0.0148,  0.3400,
-0.0910,  0.1331,  2.2702,  1.6246, -0.0808,  2.8040, -0.0229,  0.2714,
-2.8965, -2.8102, -2.4092, -0.2611, -0.7509,  1.1003, -1.1587,  2.4771,
-0.9840, -2.5090, -2.8287,  1.2677,  1.9066,  0.2307,  2.2660, -1.1267,
0.6627, -1.1031, -0.8452,  0.5336, -1.0725, -2.3834, -2.0803,  0.2847,
-1.3810,  2.3189,  1.1926, -2.3248, -1.3345, -2.0532,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000]
phi = [-1.6860, -2.1017, -3.1330,  3.0816, -1.8330,  2.5536, -2.0072, -1.5770,
-1.8331, -1.4660,  2.5851,  2.8690,  2.6332, -2.1814, -2.3828, -1.4869,
-3.0481, -0.4385,  2.1528, -1.4886,  2.1606,  0.5475, -0.1028,  0.3967,
1.3311, -1.4863,  2.0943,  2.9513, -1.1379,  1.8355,  1.9691, -1.3457,
1.7439,  0.3421, -0.2533, -1.4705, -0.9172, -2.7329, -3.1268, -0.5579,
1.3576,  2.1496,  1.9649,  0.6343,  2.1635, -1.4659,  0.9256, -1.0620,
-1.7095,  0.5678, -0.7341, -0.5942, -0.6827, -2.7838,  1.6911,  0.4873,
0.4852, -1.8838, -0.2429, -0.2303, -1.7555,  0.5094, -2.9302, -0.8538,
0.4403, -2.0446,  2.7381, -2.0512, -1.9241, -0.1962,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000]

pt = [10.8438,  4.4609,  2.7402,  2.6914,  1.9307,  1.5801,  1.1660,  1.0537,
0.8838,  0.5996,  0.5459,  0.4016,  0.2947,  0.2012,  5.9688,  4.8477,
4.7734,  4.2930,  4.1289,  3.6953,  3.3301,  3.2793,  3.2539,  3.2422,
3.1523,  3.1211,  3.0176,  3.0000,  2.3418,  2.0117,  2.0039,  1.9824,
1.8057,  1.7373,  1.5967,  1.4609,  1.4521,  1.4414,  1.3486,  1.2910,
1.2236,  1.2080,  1.2021,  1.1982,  1.1104,  1.0938,  1.0547,  1.0342,
1.0312,  1.0176,  1.0098,  0.9941,  0.8491,  0.7490,  0.7480,  0.7217,
0.5356,  0.5220,  0.5024,  0.4939,  0.4641,  0.4558,  0.4319,  0.3491,
0.3442,  0.3191,  0.3101,  0.2771,  0.2401,  0.2069,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000]

phi = [-1.6860, -2.1017, -3.1330,  3.0816, -1.8330,  2.5536, -2.0072, -1.5770,
-1.8331, -1.4660,  2.5851,  2.8690,  2.6332, -2.1814, -2.3828, -1.4869,
-3.0481, -0.4385,  2.1528, -1.4886,  2.1606,  0.5475, -0.1028,  0.3967,
1.3311, -1.4863,  2.0943,  2.9513, -1.1379,  1.8355,  1.9691, -1.3457,
1.7439,  0.3421, -0.2533, -1.4705, -0.9172, -2.7329, -3.1268, -0.5579,
1.3576,  2.1496,  1.9649,  0.6343,  2.1635, -1.4659,  0.9256, -1.0620,
-1.7095,  0.5678, -0.7341, -0.5942, -0.6827, -2.7838,  1.6911,  0.4873,
0.4852, -1.8838, -0.2429, -0.2303, -1.7555,  0.5094, -2.9302, -0.8538,
0.4403, -2.0446,  2.7381, -2.0512, -1.9241, -0.1962,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000]
energy = [ 89.6979,  37.3728,  14.6419,  22.0535,  21.0930,   8.5336,  12.7390,
8.7797,  11.2446,   6.4220,   3.7654,   2.1865,   2.4420,   1.8658,
49.9413, 180.0798, 211.2739,  55.9319,  35.3349,  32.9454,  12.3390,
6.6107,  27.4296,  25.0163,  27.9669,  97.0812,  19.9543,  23.7825,
2.6199,   2.0915,   2.0041,   2.0981,   1.8131,   1.7527,   7.8128,
3.8519,   1.4569,  11.9426,   1.3490,   1.3388,  11.1136,  10.0709,
6.7410,   1.2393,   1.4384,   1.8254,   1.8456,   6.2004,   1.5722,
6.2957,   8.5742,   1.9109,   2.9236,   0.7816,   3.6473,   1.2382,
0.6723,   0.8841,   0.7068,   0.5828,   0.7704,   2.4957,   1.7616,
0.3892,   0.7414,   1.6434,   0.5751,   1.4369,   0.5072,   0.8313,
0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
0.0000,   0.0000]
pdgid = [ 130.,   22.,  130.,   22.,    1.,  -13.,    1.,   22.,    2.,    2.,
22.,   22.,   22.,    2.,  130.,    1.,    1.,    1.,  130.,  130.,
130.,  130.,  130.,  130.,  130.,    1.,  130.,  130.,   22.,   22.,
22.,   22.,   22.,   22.,  211.,   22.,   22.,   22.,   22.,   22.,
22.,   22.,   22.,   22.,   22.,   22.,   22.,   22.,   22.,   22.,
22., -211., -211.,  211., -211., -211.,  211.,  211., -211., -211.,
-211.,  211.,  211.,  211.,  211., -211.,  211., -211.,  211., -211.,
0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]
charge = [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  1., -1., -1.,
1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1.,
0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
0.,  0.]
puppiweight = [0.0000, 0.0000, 0.9961, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8667, 0.8902,
0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.6196, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000,
1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000]
energy_log = [ 4.4964,  3.6209,  2.6839,  3.0935,  3.0489,  2.1440,  2.5447,  2.1724,
2.4199,  1.8597,  1.3259,  0.7823,  0.8928,  0.6237,  3.9108,  5.1934,
5.3532,  4.0241,  3.5649,  3.4949,  2.5128,  1.8887,  3.3116,  3.2195,
3.3310,  4.5755,  2.9934,  3.1690,  0.9631,  0.7379,  0.6952,  0.7411,
0.5951,  0.5612,  2.0558,  1.3486,  0.3763,  2.4801,  0.2994,  0.2918,
2.4082,  2.3096,  1.9082,  0.2146,  0.3635,  0.6018,  0.6128,  1.8246,
0.4525,  1.8399,  2.1488,  0.6476,  1.0728, -0.2464,  1.2940,  0.2136,
-0.3971, -0.1231, -0.3470, -0.5398, -0.2609,  0.9146,  0.5662, -0.9436,
-0.2993,  0.4967, -0.5531,  0.3625, -0.6789, -0.1848,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000]
pt_log = [ 2.3836,  1.4954,  1.0080,  0.9901,  0.6579,  0.4575,  0.1536,  0.0523,
-0.1235, -0.5115, -0.6053, -0.9123, -1.2219, -1.6036,  1.7865,  1.5785,
1.5631,  1.4570,  1.4180,  1.3071,  1.2030,  1.1876,  1.1799,  1.1762,
1.1481,  1.1382,  1.1045,  1.0986,  0.8509,  0.6990,  0.6951,  0.6843,
0.5909,  0.5523,  0.4679,  0.3791,  0.3730,  0.3656,  0.2991,  0.2554,
0.2018,  0.1890,  0.1841,  0.1809,  0.1047,  0.0896,  0.0532,  0.0336,
0.0308,  0.0174,  0.0097, -0.0059, -0.1636, -0.2890, -0.2903, -0.3262,
-0.6243, -0.6501, -0.6883, -0.7054, -0.7676, -0.7857, -0.8396, -1.0523,
-1.0664, -1.1423, -1.1710, -1.2834, -1.4266, -1.5755,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000]

mask = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]


pf_points = [eta, phi]
pf_features = [pt, eta, phi, energy, pdgid, charge, puppiweight, energy_log, pt_log]
pf_mask = [mask]

pf_points = np.array([pf_points]).astype(np.float32)
pf_features = np.array([pf_features]).astype(np.float32)
pf_mask = np.array([pf_mask]).astype(np.float32)

dict_inputs = {'pf_points': pf_points,'pf_features': pf_features,'pf_mask': pf_mask}

pf_points = torch.from_numpy(pf_points)
pf_features = torch.from_numpy(pf_features)
pf_mask = torch.from_numpy(pf_mask)

inputs = [pf_points, pf_features, pf_mask]

score = model(*inputs)
score = torch.softmax(score, dim=1).detach().cpu().numpy()
score_onxx = ort_sess.run([], dict_inputs)[0]
print("test", score, score_onxx)
