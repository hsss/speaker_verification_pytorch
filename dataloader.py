import pathlib
import numpy as np

import torch

import pickle

from torch.utils.data import Dataset, DataLoader

from mfcc_IO import *


class VoxCeleb_with_mask(Dataset):
	def __init__(self, lines, spk_dic, num_frames):
		self.lines = lines
		self.spk_dic = spk_dic
		self.num_frames = num_frames


	def __getitem__(self, index):
		line = self.lines[index]

		tmp = line.strip().split(' ')
		utt = tmp[0] 
			
		pointer = int (tmp[1].split(':')[1])
		src = open(tmp[1].split(':')[0], 'rb')
		src.seek(pointer)
		spec = np.array(read_kaldi_mfcc(src), np.float32)
		src.close()
		
		spk = utt.split('/')[0]
		spk = self.spk_dic[spk]			
		ans = spk
					
		while spec.shape[0] < self.num_frames:
			spec = np.concatenate([spec, spec])
			
		margin = int((spec.shape[0] - self.num_frames)/2)
		
		if margin == 0:
			st_idx = 0
		else:
			st_idx = np.random.randint(0, margin)
			
		ed_idx = st_idx + self.num_frames
		
		data = spec[st_idx:ed_idx,:]
		data = data.reshape((data.shape[0], data.shape[1], 1))

		ans2 = list(range(len(self.spk_dic)))
		del ans2[ans]

		
		return np.transpose(data, (2,0,1 )), [ ans, np.array(ans2,np.int32)]

	def __len__(self):
		return len(self.lines)


class VoxCeleb_test(Dataset):
	def __init__(self, val_lines, val_key):
		self.val_lines = val_lines
		self.val_key = val_key
		

	def __getitem__(self, index):
		line = self.val_lines[index]

		tmp = line.strip().split(' ')
		utt = tmp[0] 
			
		pointer = int (tmp[1].split(':')[1])
		src = open(tmp[1].split(':')[0], 'rb')
		src.seek(pointer)
		spec = read_kaldi_mfcc(src)
		src.close()
		spec = spec.reshape((spec.shape[0], spec.shape[1], 1))
		
		return np.transpose(spec, (2,0,1 )), self.val_key[index]

	def __len__(self):
		return len(self.val_lines)

def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_loader(config):
	batch_size = config['batch_size']
	num_workers = config['num_workers']
	use_gpu = config['use_gpu']

	voxceleb2_meta = open('scp/vox2_meta.csv').readlines()[1:]
		
	voxceleb2_spk_split = {}
	for line in voxceleb2_meta:
		#'id09263 ,n009263 ,m ,dev \n'
		tmp = line.split(' ,')
		
		voxceleb2_spk_split[tmp[0]] = tmp[3]
	
	
	voxceleb2_val_lines = []
	voxceleb2_val_key = []
	voxceleb2_tr_lines = []
	

	voxceleb2_lines = open('scp/fbank_voxceleb2.scp', 'r').readlines()
	spk_dic = {}		
	tr_lines = voxceleb2_lines
	for line in tr_lines:
		tmp = line.strip().split(' ')
		utt = tmp[0]
		
		spk = utt.split('/')[0]
		
		if 'test' in voxceleb2_spk_split[spk] :
			voxceleb2_val_lines.append(line)
			voxceleb2_val_key.append(utt)
		else:
			voxceleb2_tr_lines.append(line)
			if spk not in spk_dic:
				spk_dic[spk] = len(spk_dic)
	
	print('Number of tr spks', len(spk_dic))
	
	config['n_classes'] = len(spk_dic)

	train_dataset =  VoxCeleb_with_mask(voxceleb2_tr_lines, spk_dic, config['num_frames'])
	test_dataset = VoxCeleb_test(voxceleb2_val_lines, voxceleb2_val_key)
	
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=use_gpu,
		drop_last=True,
		worker_init_fn=worker_init_fn,
	)
	
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=1,
		num_workers=num_workers,
		shuffle=False,
		pin_memory=use_gpu,
		drop_last=False,
	)

	return train_loader, test_loader
