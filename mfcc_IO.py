import struct
import numpy as np

def read_kaldi_mfcc(f):
	'''
	header length: 15 byte
	TTTTxxllxxxdxxx
	T: data type
	x: unknown
	l: length of feature
	d: dimension of feature
	'''	

	f.seek(f.tell() + 6)

	length = struct.unpack('i',f.read(4))[0]
	if length < 0:
		return False
	
	f.seek(f.tell() + 1)
	dim = struct.unpack('h',f.read(2))[0]

	num_elements = length * dim
	
	f.seek(f.tell() + 2)
	feats = struct.unpack('%df'%(num_elements), f.read(num_elements * 4))
	feats = np.array(feats).reshape(length, dim)
	
	return feats
	
def read_kaldi_ivector(f):
	'''
	header length: 10 byte
	TTTTxxllxx
	T: data type
	x: unknown
	l: length of feature
	'''	
	f.seek(f.tell() + 6)
	length = struct.unpack('i',f.read(4))[0]
	
	num_elements = length 
	
	#f.seek(f.tell() + 2)
	feats = struct.unpack('%df'%(num_elements), f.read(num_elements * 4))
	feats = np.array(feats)
	
	return feats 
	
def read_kaldi_vad(f):
	'''
	header length: 10 byte
	TTTTxxllll
	T: data type
	x: unknown
	l: length of feature
	
	for i in xrange(10):
		print f.read(1)
	asdasda()
	'''
	f.seek(f.tell() + 6)
	length = struct.unpack('i',f.read(4))[0]

	if length < 0:
		return False
	
	num_elements = length 
	
	#f.seek(f.tell() + 2)
	feats = struct.unpack('%df'%(num_elements), f.read(num_elements * 4))
	feats = np.array(feats)
	
	return feats 
def write_kaldi_mfcc(f, feats):
	f.write('\x00')
	f.write('BFM ')
	f.write('\x04')
	
	length = feats.shape[0]
	dim = feats.shape[1]
	num_elements = length * dim
	f.write(struct.pack('h',length))
	f.write('\x00\x00\x04')
	f.write('%c'%(dim))
	f.write('\x00\x00\x00')
	f.write(struct.pack('%df'%(num_elements), *feats.reshape(1,num_elements)[0].tolist()))
	
	return f
	
def write_kaldi_vad(f, feats):
	f.write('\x00')
	f.write('BFV ')
	f.write('\x04')
	
	length = feats.shape[0]
	f.write(struct.pack('l',length))
	#f.write('\x00\x00')
	f.write(struct.pack('%df'%(length), *feats.tolist()))
	
	return f
	
def write_kaldi_ivector(f, feats):
	f.write('\x00'.encode('utf-8'))
	f.write('BFV '.encode('utf-8'))
	f.write('\x04'.encode('utf-8'))
	
	length = feats.shape[0]
	f.write(struct.pack('l',length))
	#f.write('\x00\x00')
	f.write(struct.pack('%df'%(length), *feats.tolist()))
	
	return f





