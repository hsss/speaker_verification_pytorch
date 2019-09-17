from multiprocessing import Process
from os import system
from os.path import splitext, basename
import numpy as np

_num_process = 2

def cmd_process(pid, cmd):
	print ('Procsee %d started'%(pid))
	system(cmd)
	print ('Procsee %d ended'%(pid))
	 

def split_scp(feat_scp_path, tmpfile_dir):

	f_feat = open(feat_scp_path, 'r')
	feat_lines = f_feat.readlines()
	f_feat.close()

	
	tmp = list(range(len(feat_lines)))
	np.random.shuffle(tmp)

	feat_lines = np.take(feat_lines, tmp)
	
	feat_tmp = splitext(basename(feat_scp_path))
	
	num_utt_process = int(len(feat_lines)/_num_process)
	
	# split scp
	for i in range(_num_process):
		feat_f = open('%s/%s_%d%s'%(tmpfile_dir, feat_tmp[0], i, feat_tmp[1]), 'w')
	
		for line in feat_lines[i*num_utt_process: (i+1)*num_utt_process]:
			feat_f.write(line)
		
		feat_f.close()
	
	feat_f = open('%s/%s_%d%s'%(tmpfile_dir, feat_tmp[0], _num_process-1, feat_tmp[1]), 'a')
	for line in feat_lines[_num_process * num_utt_process:]:
		feat_f.write(line)
	feat_f.close()
	
	return ['%s/%s'%(tmpfile_dir, feat_tmp[0]), feat_tmp[1]]

		
if __name__ == '__main__':
	
	
	wav_tmp = split_scp('scp/wav_voxceleb1.scp', 'tmp')
	#wav_tmp = split_scp('scp/wav_voxceleb2.scp', 'tmp')
	process_list = []
	i = 0
	
	for i in range(_num_process):
		wav_scp = '%s_%d%s'%(wav_tmp[0], i, wav_tmp[1])
		cmd = (	'compute-fbank-feats --verbose=1 --num-mel-bins=64 --sample-frequency=16000 --window-type=hamming scp:%s ark:- |'
				'apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark,scp:data/%s_%d.ark,tmp/%s_%d.scp'
				%(wav_scp, basename(wav_tmp[0]).replace('wav', 'fbank'),i, basename(wav_tmp[0]).replace('wav', 'fbank'),i))
		
		
		
		process_list.append(Process(target = cmd_process, args = (i,cmd)))
	
	for pid in range(_num_process):
		process_list[pid].start()
	for pid in range(_num_process):
		process_list[pid].join()
	
	f = open('scp/fbank_voxceleb1.scp', 'w')
	#f = open('scp/fbank_voxceleb2.scp', 'w')
	
	for i in range(_num_process):
		
		lines = open('tmp/%s_%d.scp'%(basename(wav_tmp[0]).replace('wav', 'fbank'), i), 'r').readlines()
		
		for line in lines:
			f.write(line)
			
	f.close()
