#!/usr/bin/env python
import os

import yaml
import shutil
import importlib

import collections
import pathlib
import time

import argparse
import numpy as np
import random

from EER import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import get_loader
import utils
from utils import str2bool, AverageMeter, ts_loss, cross_entropy_loss, onehot_encoding

import zipfile

torch.backends.cudnn.benchmark = True
global_step = 0


def zipdir(path, ziph):
	for root, dirs, files in os.walk(path):
		for file in files:
			fn, ext = os.path.splitext(file)
			if ext != '.py': continue

			ziph.write(os.path.join(root, file))

def train(epoch, model, optimizer, scheduler, train_criterion, train_loader, config, f_results):
	global global_step
	
	device = torch.device(config['device'])

	model.train()
	loss_cos = torch.nn.CosineEmbeddingLoss()
	
	negative_mask = torch.ones((config['n_classes'],config['n_classes']), dtype=torch.float32) - nn.init.eye_(torch.empty(config['n_classes'],config['n_classes']))					
	negative_mask = negative_mask.to(device, dtype=torch.float)
		

	loss_meter = AverageMeter()
	accuracy_meter = AverageMeter()
	start = time.time()
	for step, (data, targets) in enumerate(train_loader):
		global_step += 1
		 
		if config['scheduler'] in ['multistep', 'sgdr']:
			scheduler.step(epoch - 1)
		elif config['scheduler'] in ['cosine', 'keras_decay']:
			scheduler.step()

		if torch.cuda.device_count() == 1:
			data = data.to(device, dtype=torch.float)
			
		optimizer.zero_grad()
				
		ans = targets[0]
		ans2 = targets[1]
		
		ans = ans.to(device, dtype=torch.long)
		ans2 = ans2.to(device, dtype=torch.long)
		
		H_loss, outputs = model(data, ans, ans2, False)

		
		if torch.cuda.device_count() == 1:
			output_weight = model.fc_output.weight
		else:
			output_weight = model.module.fc_output.weight
		
		norm = torch.norm(output_weight, dim=1, keepdim=True) / (5. ** 0.5)
		normed_weight = torch.div(output_weight, norm ) 

		inner = torch.mm(normed_weight, normed_weight.t())			
		BC_loss = torch.log(torch.exp( (inner * negative_mask) ** 2. ).mean())
		H_loss = H_loss.mean()
		CCE_loss = train_criterion(outputs, ans)
		loss =  H_loss + CCE_loss + (BC_loss *config['BC_weight'])
		

		total_loss = loss 
	
			
		total_loss.backward()
		if 'gradient_clip' in config.keys():
			torch.nn.utils.clip_grad_norm_(model.parameters(),
											config['gradient_clip'])
		optimizer.step()
		
		loss_ = loss.item()
		
		num = data.size(0)
		accuracy = utils.accuracy(outputs, ans)[0].item()

		loss_meter.update(loss_, num)
		accuracy_meter.update(accuracy, num)

		if step % 100 == 0:
			print('Epoch {} Step {}/{} '
						'Loss {:.4f} ({:.4f}) '
						'Accuracy {:.4f} ({:.4f})'.format(
							epoch,
							step,
							len(train_loader),
							loss_meter.val,
							loss_meter.avg,
							accuracy_meter.val,
							accuracy_meter.avg,
						))
			

	elapsed = time.time() - start
	f_results.write('Epoch {} Step {}/{} '
						'Loss {:.4f} ({:.4f}) '
						'Accuracy {:.4f} ({:.4f})  '.format(
							epoch,
							step,
							len(train_loader),
							loss_meter.val,
							loss_meter.avg,
							accuracy_meter.val,
							accuracy_meter.avg,
						))
	f_results.write('Elapsed {:.2f}\n'.format(elapsed))
	print('Elapsed {:.2f}'.format(elapsed))

	train_log = collections.OrderedDict({
		'epoch':
		epoch,
		'train':
		collections.OrderedDict({
			'loss': loss_meter.avg,
			'accuracy': accuracy_meter.avg,
			'time': elapsed,
		}),
	})
	return train_log


def test(epoch, model, tst_lines, test_loader, config, f_results):
	_trial_str = ['imposter', 'trueSpeaker']

	device = torch.device(config['device'])
	target_dummy = torch.ones((config['batch_size'], )).to(device, dtype=torch.long)
	model.eval()
		
	loss_meter = AverageMeter()
	correct_meter = AverageMeter()
	start = time.time()
	with torch.no_grad():
		
		e_dic = {}
				
		for	data, targets in test_loader:
			data = data.to(device, dtype=torch.float)
			code = model(data, data, data,  True)
			e_dic[targets[0]] = np.array(code.cpu(), np.float32)[0]

		score_list = []
	
		for line in tst_lines:
			tmp = line.strip().split(' ')
			score = np.dot(e_dic[tmp[1]], e_dic[tmp[2]])
			score_list.append([_trial_str[int(tmp[0])],score])
		threshold = calculateEER(score_list)
		FARate, FRRate, correctNum, wrongNum = getErrorRate(score_list, threshold = threshold)
		val_eer = np.mean([FARate, FRRate])
		
		print('Epoch {}  EER {:.4f}'.format(epoch,  val_eer))
		f_results.write('Epoch {}  EER {:.4f} '.format(epoch,  val_eer))

		elapsed = time.time() - start

		print('Elapsed {:.2f}'.format(elapsed))
		f_results.write('Elapsed {:.2f}\n'.format(elapsed))

	return val_eer

	
def update_state(state, epoch, val_eer, model, optimizer):
	state['state_dict'] = model.state_dict()
	state['optimizer'] = optimizer.state_dict()
	state['epoch'] = epoch
	state['val_eer'] = val_eer

	# update best accuracy
	if val_eer < state['best_val_eer']:
		state['best_val_eer'] = val_eer
		state['best_epoch'] = epoch

	return state


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str)
	args = parser.parse_args()
		
	with open(args.config, 'r') as f_yaml:
		config = yaml.load(f_yaml)
	
	outdir = pathlib.Path(config['outdir'])
	outdir.mkdir(exist_ok=True, parents=True)

	shutil.copyfile(args.config, str(outdir) + '/config.yaml')

	zipf = zipfile.ZipFile(str(outdir) + '/codes.zip', 'w', zipfile.ZIP_DEFLATED)
	zipdir('./', zipf)
	zipf.close()
	
	f_results = open(str(outdir) + '/results.txt', 'w', buffering = 1)

	seed = config['seed']

	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	epoch_seeds = np.random.randint(
		np.iinfo(np.int32).max // 2, size=config['epochs'])
		
	train_loader, test_loader = get_loader(config)
	
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

	train_criterion = nn.CrossEntropyLoss(reduction='mean')
	test_criterion = nn.CrossEntropyLoss(reduction='mean')
	
	#params = model.parameters()
	
	params = [
		{
			'params': [
				param for name, param in model.named_parameters()
				if 'bn' not in name
			]
		},
		{
			'params': [
				param for name, param in model.named_parameters()
				if 'bn' in name
			],
			'weight_decay':
			0
		},
	]

	counter = 0
 
	config['steps_per_epoch'] = len(train_loader)
	optimizer, scheduler = utils.create_optimizer(params, config)
	
	voxceleb2_val = open('scp/voxceleb2_val.txt', 'r').readlines()
	
	# run test before start training
	if config['test_first']:
		test(0, model, voxceleb2_val, test_loader, config, f_results)
	
	state = {
		'config': config,
		'state_dict': None,
		'optimizer': None,
		'epoch': 0,
		'val_eer': 50.,
		'best_val_eer': 50.,
		'best_epoch': 0,
	}
	epoch_logs = []
	for epoch, seed in zip(range(1, config['epochs'] + 1), epoch_seeds):
		np.random.seed(seed)
		
		train_log = train(epoch, model, optimizer, scheduler, train_criterion, train_loader, config, f_results)

		epoch_log = train_log.copy()
		
		val_eer = test(epoch, model, voxceleb2_val, test_loader, config, f_results)
							
		epoch_logs.append(epoch_log)
	

		# update state dictionary
		state = update_state(state, epoch, val_eer, model, optimizer)

		# save model
		utils.save_checkpoint(state, outdir)
		

if __name__ == '__main__':
	main()
