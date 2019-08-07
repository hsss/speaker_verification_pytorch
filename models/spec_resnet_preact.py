import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
	if isinstance(module, nn.Conv2d):
		nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
	elif isinstance(module, nn.BatchNorm2d):
		module.weight.data.fill_(1)
		module.bias.data.zero_()
	elif isinstance(module, nn.Linear):
		module.bias.data.zero_()


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride, preact=False):
		super(BasicBlock, self).__init__()

		self._preact = preact

		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv1 = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=3,
			stride=stride,  # downsample with first conv
			padding=1,
			bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(
			out_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=False)

		self.shortcut = nn.Sequential()
		if in_channels != out_channels:
			self.shortcut.add_module(
				'conv',
				nn.Conv2d(
					in_channels,
					out_channels,
					kernel_size=1,
					stride=stride,  # downsample
					padding=0,
					bias=False))

	def forward(self, x):
		if self._preact:
			x = F.relu(
				self.bn1(x), inplace=True)  # shortcut after preactivation
			y = self.conv1(x)
		else:
			# preactivation only for residual path
			y = self.bn1(x)

			y = F.relu(y, inplace=True)
			y = self.conv1(y)

		y = F.relu(self.bn2(y), inplace=True)
		y = self.conv2(y)

		y += self.shortcut(x)
		return y



class Network(nn.Module):
	def __init__(self, config):
		super(Network, self).__init__()
		self.config = config
		input_shape = config['input_shape']
		n_classes = config['n_classes']

		base_channels = config['base_channels']
		n_stages = config['n_stages']

		stride_per_stage = [(1, 1), (2, 2), (2, 2), (2, 2)]
		n_blocks_per_stage = [3, 4, 6, 4]
		preact_stage = [True, False, False, False, False, False]
		
		

		block = BasicBlock
		
		n_channels = [
			base_channels,
			base_channels, 
			base_channels * 2, 
			base_channels * 4, 
			base_channels * 8, 
			base_channels * 16, 
		]
		
		self.conv = nn.Conv2d(
			input_shape[1],
			n_channels[0],
			kernel_size=(7, 7),
			stride=(1,2),
			padding=1,
			bias=False)
		
		self.stage_list = []

		self.stage1 = self._make_stage(
						n_channels[0],
						n_channels[1],
						n_blocks_per_stage[0],
						block,
						stride=stride_per_stage[0],
						preact=preact_stage[0])

		self.stage2 = self._make_stage(
						n_channels[1],
						n_channels[2],
						n_blocks_per_stage[1],
						block,
						stride=stride_per_stage[1],
						preact=preact_stage[1])
		
		self.stage3 = self._make_stage(
						n_channels[2],
						n_channels[3],
						n_blocks_per_stage[2],
						block,
						stride=stride_per_stage[2],
						preact=preact_stage[2])

		self.stage4 = self._make_stage(
						n_channels[3],
						n_channels[4],
						n_blocks_per_stage[3],
						block,
						stride=stride_per_stage[3],
						preact=preact_stage[3])

		self.bn = nn.BatchNorm2d(n_channels[4])

		# compute conv feature size
		with torch.no_grad():
			self.feature_size = self._forward_conv(
				torch.zeros(*input_shape)).view(-1).shape[0]

		self.fc_code = nn.Linear(self.feature_size, config['code_dim'])
		self.fc_output = nn.Linear(config['code_dim'], n_classes)
	

		# initialize weights
		self.apply(initialize_weights)

	def _make_stage(self, in_channels, out_channels, n_blocks, block, stride,
					preact):
		stage = nn.Sequential()
		for index in range(n_blocks):
			block_name = 'block{}'.format(index + 1)
			if index == 0:
				stage.add_module(block_name,
								 block(
									 in_channels,
									 out_channels,
									 stride=stride,
									 preact=preact))
			else:
				stage.add_module(block_name,
								 block(
									 out_channels,
									 out_channels,
									 stride=1,
									 preact=False))
		return stage

	def _forward_conv(self, x):
		
	
		x = self.conv(x)
		
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		
		x = F.relu(
			self.bn(x),
			inplace=True) 
		x_a = F.adaptive_avg_pool2d(x, output_size=1)
		x_m = F.adaptive_max_pool2d(x, output_size=1)
		
		return x_a + x_m

	def forward(self, x):
		x = self._forward_conv(x)
		code = self.fc_code(x.view(x.size(0), -1))
				
		if self.config['code_norm']:
			norm = code.norm(p=2, dim=1, keepdim=True)
			code = torch.div(code, norm / self.config['norm_s'])

		x = self.fc_output(code)
		
		return x, code
