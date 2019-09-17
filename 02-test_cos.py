#!/usr/bin/env python
import os

import yaml

import importlib

import pathlib
import time

import argparse
import numpy as np
import random

from EER import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import VoxCeleb_test
import utils
from utils import str2bool, AverageMeter, ts_loss, cross_entropy_loss, onehot_encoding

_trial_str = ['imposter', 'trueSpeaker']

def get_eer(tst_lines, e_dic):
	score_list = []
	
	for line in tst_lines:
		tmp = line.strip().split(' ')
		score = np.dot(e_dic[tmp[1]], e_dic[tmp[2]])
		score_list.append([_trial_str[int(tmp[0])],score])
	threshold = calculateEER(score_list)
	FARate, FRRate, correctNum, wrongNum = getErrorRate(score_list, threshold = threshold)
	tst_eer = np.mean([FARate, FRRate])
		
	return tst_eer

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str)
	args = parser.parse_args()
		
	with open(args.config, 'r') as f_yaml:
		config = yaml.load(f_yaml)

	voxceleb1_all_lines = open('scp/fbank_voxceleb1.scp', 'r').readlines()

	voxceleb1_val_lines = []
	voxceleb1_val_key = []

	for line in voxceleb1_all_lines:
		tmp = line.strip().split(' ')
		utt = tmp[0]
		spk = utt.split('/')[0]
		
		voxceleb1_val_lines.append(line)
		voxceleb1_val_key.append(utt)
	
	num_workers = config['num_workers']
	use_gpu = config['use_gpu']
		
	test_dataset = VoxCeleb_test(voxceleb1_val_lines, voxceleb1_val_key)
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=1,
		num_workers=num_workers,
		shuffle=False,
		pin_memory=use_gpu,
		drop_last=False,
	)
	
	module = importlib.import_module('models.{}'.format(config['arch']))
	Network = getattr(module, 'Network')
	model = Network(config)

	n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
	print('n_params: {}'.format(n_params))

	device = torch.device(config['device'])
	
	if config['load_weights']:
		state_dict = torch.load(config['model_path'])['state_dict']
		model.load_state_dict(state_dict, strict=True)
		
	if device.type == 'cuda' and torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	
	model.to(device)	
	model.eval()
	
	print(config['model_path'])

	with torch.no_grad():
		
		e_dic = {}
				
		for	data, targets in test_loader:
			data = data.to(device, dtype=torch.float)
			
			print(targets[0])

			code = model(data, data,  data, True)

			e_dic[targets[0]] = np.array(code.cpu(), np.float32)[0]

	
	print(config['model_path'])
	tst_lines = open('scp/voxceleb1_test.txt', 'r').readlines()	
	tst_eer = get_eer(tst_lines, e_dic)
	print('40-tst EER: %f'%(tst_eer))

	tst_lines = open('scp/Vox1-E.scp', 'r').readlines()	
	tst_eer = get_eer(tst_lines, e_dic)
	print('E-tst EER: %f'%(tst_eer))

	tst_lines = open('scp/Vox1-H.scp', 'r').readlines()	
	tst_eer = get_eer(tst_lines, e_dic)
	print('H-tst EER: %f'%(tst_eer))
	


if __name__ == '__main__':
	main()
