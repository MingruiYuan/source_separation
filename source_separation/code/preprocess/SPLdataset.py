import os
import random as rd
import torch
# import cv2
import numpy as np
from scipy.io import wavfile
import scipy.ndimage as sn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import librosa.core as lc
from .STFT import *

# this parameters should < min(size1, size2)
# 这个参数表明在每种乐器中选取几个片段。
number_of_episode_in_use = 1 # must < 21!!
episode_size_log = [0]*number_of_episode_in_use

OP_POINT = 65280
imgnum = 14.2106

def combination_sequence(in1, in2):
	num2ins = ['accordion','acoustic_guitar','cello','flute','saxophone','trumpet','violin','xylophone']

	cbsq = []
	#for i in in1:
	#	for j in in2:
	#		if i != j:
	#			cbsq.append([num2ins[i], num2ins[j]])
	for i in range(len(in1)):
		for j in range(len(in2)):
			if (j>=i) and (in1[i] != in2[j]):
				cbsq.append([num2ins[in1[i]], num2ins[in2[j]]])
	#print(len(cbsq))
			
	return cbsq

class SPLdataset(Dataset):
	"""
	Sources separation and localization Dataset.
	"""

	def __init__(self, ins1, ins2, val=False):
		"""
		Arguments:
		ins1, ins2: list of number, mapped to name of instruments according to num2ins.
		"""
		# self.ins1 = ins1
		# self.ins2 = ins2
		# self.num2ins = ['accordion','acoustic_guitar','cello','flute','saxophone','trumpet','violin','xylophone']
		# change root directory by yourself
		self.val = val
		# counters
		self.seq_num = 0
		self.seq = combination_sequence(ins1, ins2)

		if not val:
			self.chosen = {"accordion":np.zeros((51,)), "acoustic_guitar":np.zeros((48,)), "cello":np.zeros((51,)), "flute":np.zeros((43,)), "saxophone":np.zeros((21,)), "trumpet":np.zeros((38,)), "violin":np.zeros((45,)), "xylophone":np.zeros((44,))}
			self.root_dir = '../../dataset'#'/home/zhangwq01/YuanMingrui/dataset' #'/Volumes/WWPOKENET/homework/dataset'
		else:
			self.chosen = {"accordion":np.zeros((56,)), "acoustic_guitar":np.zeros((55,)), "cello":np.zeros((57,)), "flute":np.zeros((48,)), "saxophone":np.zeros((24,)), "trumpet":np.zeros((43,)), "violin":np.zeros((50,)), "xylophone":np.zeros((49,))}
			self.root_dir = '../../MUSIC_dataset-master'

	def __getitem__(self, idx):
		pass

	def __len__(self):
		pass

	def sequence_modification(self, in1, in2, list1, list2, num_in_use):
		final_list1 = [0]*num_in_use
		final_list2 = [0]*num_in_use
		for k in range(num_in_use):
			if self.chosen[in1].all():
				self.chosen[in1] = np.zeros((len(list1),))
			if self.chosen[in2].all():
				self.chosen[in2] = np.zeros((len(list2),))

			unchs1 = np.where(self.chosen[in1] == 0)[0]
			np.random.shuffle(unchs1)
			unchs2 = np.where(self.chosen[in2] == 0)[0]
			np.random.shuffle(unchs2)

			if self.chosen[in1][int(list1[k][:-4])-1] == 1:
				final_list1[k] = unchs1[0] + 1
				self.chosen[in1][unchs1[0]] = 1
			else:
				final_list1[k] = int(list1[k][:-4])
				self.chosen[in1][int(list1[k][:-4])-1] = 1

			if self.chosen[in2][int(list2[k][:-4])-1] == 1:
				final_list2[k] = unchs2[0] + 1
				self.chosen[in2][unchs2[0]] = 1
			else:
				final_list2[k] = int(list2[k][:-4])
				self.chosen[in2][int(list2[k][:-4])-1] = 1

		#print(final_list1)
		#print(final_list2)

		return final_list1, final_list2

	def get_image(self, in1, in2, fnls1, fnls2):
		"""
		in1,in2: string, name of instruments.
		image1, image2: list of 4-D torch tensor, N*C*H*W. 
			N--number of frames in each episode(according to the shorter one).
			C--channels = 3.
			H, W--size of images.
		Each tensor is float [0,1].
		"""		
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		normalize = transforms.Compose(
    	[transforms.Resize(256),
    	 transforms.CenterCrop(224),
    	 transforms.ToTensor(),
    	 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    	])
		dir1 = self.root_dir + '/images/solo/' + in1
		dir2 = self.root_dir + '/images/solo/' + in2
		# list1 = os.listdir(dir1)
		# list2 = os.listdir(dir2)
		# size1 = len(list1)
		# size2 = len(list2)

		# allocate memory in advance
		# num_in_use = min(min(size1,size2), number_of_episode_in_use)
		num_in_use = number_of_episode_in_use #NOTICE
		image1 = torch.Tensor(sum(episode_size_log)-2*num_in_use, 3, 3, 224, 224)#.cuda()
		image2 = torch.Tensor(sum(episode_size_log)-2*num_in_use, 3, 3, 224, 224)#.cuda()
		counter = 0

		for k in range(num_in_use):
			imagefile1 = dir1 + '/' + str(fnls1[k])
			imagefile2 = dir2 + '/' + str(fnls2[k])
			all_img1 = os.listdir(imagefile1)
			all_img1.sort(key = lambda x:int(x[:-4]))
			all_img2 = os.listdir(imagefile2)
			all_img2.sort(key = lambda x:int(x[:-4]))
			image_num1 = len(all_img1)
			#print(image_num1)
			image_num2 = len(all_img2)
			#print(image_num2)

			# initialize tensors
			# img_per_episode1 = torch.Tensor(sum(episode_size_log), 3, 3, 224, 224)
			# img_per_episode2 = torch.Tensor(sum(episode_size_log), 3, 3, 224, 224)
			# crop the longer episode
			for p in range(episode_size_log[k]):
				if p > 0 and p < episode_size_log[k]-1:
					#print(episode_size_log[k])
					#print(p)
					x1 = round(p*imgnum)+1
					x3 = round((p+1)*imgnum)-1
					x2 = (x1+x3) // 2
					#print(x3)
					# rescale images to [0,1]
					image1[counter][0] = normalize(Image.open(imagefile1 + '/' + all_img1[x1]))  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x1]),(224,224))/255, (2,0,1))))
					
					image1[counter][1] = normalize(Image.open(imagefile1 + '/' + all_img1[x2]))  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x2]),(224,224))/255, (2,0,1))))
					
					image1[counter][2] = normalize(Image.open(imagefile1 + '/' + all_img1[x3]))  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x3]),(224,224))/255, (2,0,1))))
					
					image2[counter][0] = normalize(Image.open(imagefile2 + '/' + all_img1[x1]))  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile2 + '/' + all_img2[x1]),(224,224))/255, (2,0,1))))
					
					image2[counter][1] = normalize(Image.open(imagefile2 + '/' + all_img1[x2]))  #normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile2 + '/' + all_img2[x2]),(224,224))/255, (2,0,1))))
					
					image2[counter][2] = normalize(Image.open(imagefile2 + '/' + all_img1[x3]))  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile2 + '/' + all_img2[x3]),(224,224))/255, (2,0,1))))
					
					counter = counter + 1

		return image1, image2	

	def get_audio(self, in1, in2):
		"""
		in1,in2: string, name of instruments.
		audio1, audio2: list of 1-D numpy array, 1st audio and 2nd audio, cropped according to the shorter one.
		audio_combined: list of 1-D numpy array, mixed audio, with the same size of the shorter one.
		"""
		dir1 = self.root_dir + '/audios/solo/' + in1
		dir2 = self.root_dir + '/audios/solo/' + in2
		list1 = os.listdir(dir1)
		list2 = os.listdir(dir2)
		#print(len(list1))
		#print(len(list2))
		# size1 = len(os.listdir(dir1))
		# size2 = len(os.listdir(dir2))
		# shuffle the order of images and audios
		#rd.shuffle(list1)
		#rd.shuffle(list2)

		# allocate memory in advance
		# num_in_use = min(min(size1,size2), number_of_episode_in_use)
		num_in_use = number_of_episode_in_use ## NOTICE
		# audio1 = [0]*num_in_use
		# audio2 = [0]*num_in_use
		# audio_combined = [0]*num_in_use
		fnls1, fnls2 = self.sequence_modification(in1, in2, list1, list2, num_in_use)

		for k in range(num_in_use):
			audiofile1 = dir1 + '/' + str(fnls1[k]) + '.wav'
			fs1, tp1 = wavfile.read(audiofile1)
			audiofile2 = dir2 + '/' + str(fnls2[k]) + '.wav'
			fs2, tp2 = wavfile.read(audiofile2)
			if not self.val:
				tp1 = sn.zoom(tp1.astype('float64'), 11025/fs1, order=2)
				tp2 = sn.zoom(tp2.astype('float64'), 11025/fs2, order=2)
			else:
				tp1, __ = lc.load(path=audiofile1, sr=fs1, mono=True)
				tp2, __ = lc.load(path=audiofile2, sr=fs2, mono=True)
				tp1 = sn.zoom(tp1.astype('float64'), 11025/fs1, order=2)
				tp2 = sn.zoom(tp2.astype('float64'), 11025/fs2, order=2)
			#tp1 = lc.resample(tp1.astype('float32'), 44100, 11025)
			#tp2 = lc.resample(tp2.astype('float32'), 44100, 11025)
			if self.val:
                		tp1 *= 31009.2 
                		tp2 *= 31009.2
			curr_len = min(len(tp1),len(tp2))
			tp1 = tp1[:curr_len]
			tp2 = tp2[:curr_len]
			#print(curr_len)
			curr_N = curr_len // OP_POINT
			#print(curr_N)
			episode_size_log[k] = curr_N
		#print(episode_size_log)

		audio_spectrum1 = torch.Tensor(sum(episode_size_log)-2*num_in_use, 1, 256, 256)#.cuda()
		audio_spectrum2 = torch.Tensor(sum(episode_size_log)-2*num_in_use, 1, 256, 256)#.cuda()
		audio_spectrum_cb = torch.Tensor(sum(episode_size_log) - 2 * num_in_use, 1, 256, 256)  #.cuda()
		if torch.cuda.is_available():
			audio_spectrum1 = audio_spectrum1.cuda()
			audio_spectrum2 = audio_spectrum2.cuda()
			audio_spectrum_cb = audio_spectrum_cb.cuda()
		audio_phase1 = np.zeros((sum(episode_size_log)-2*num_in_use, 512, 256))
		audio_phase2 = np.zeros((sum(episode_size_log)-2*num_in_use, 512, 256))
		audio_phase_cb = np.zeros((sum(episode_size_log)-2*num_in_use, 512, 256))

		counter = 0
		for k in range(num_in_use):
			audiofile1 = dir1 + '/' + str(fnls1[k]) + '.wav'
			#print(audiofile1)
			fs, tp1 = wavfile.read(audiofile1)
			audiofile2 = dir2 + '/' + str(fnls2[k]) + '.wav'
			#print(audiofile2)
			fs, tp2 = wavfile.read(audiofile2)
			# crop the longer audio
			
			# episode_size_log[k] = curr_N

			if not self.val:
				tp1 = sn.zoom(tp1.astype('float64'), 11025/fs1, order=2)
				tp2 = sn.zoom(tp2.astype('float64'), 11025/fs2, order=2)
			else:
				tp1, __ = lc.load(path=audiofile1, sr=fs1, mono=True)
				tp2, __ = lc.load(path=audiofile2, sr=fs2, mono=True)
				tp1 = sn.zoom(tp1.astype('float64'), 11025/fs1, order=2)
				tp2 = sn.zoom(tp2.astype('float64'), 11025/fs2, order=2)
			# tp1 = lc.resample(tp1.astype('float32'), 44100, 11025)
			#tp2 = lc.resample(tp2.astype('float32'), 44100, 11025)
			if self.val:
				tp1 *= 31009.2 
				tp2 *= 31009.2
			curr_len = min(len(tp1),len(tp2))
			tp1 = tp1[:curr_len]
			tp2 = tp2[:curr_len]
			#print(curr_len)
			curr_N = episode_size_log[k]
			# # Need rescaling of amplitude ?
			# audio1[k] = tp1
			# audio2[k] = tp2
			# audio_combined[k] = np.add(tp1,tp2)
			#print(curr_N)
			for p in range(curr_N):
				# discard the first and last part in each audio.
				if p > 0 and p < curr_N-1:
					#print(p)
					part1 = tp1[p*OP_POINT:(p+1)*OP_POINT]
					#print(part1.shape)
					part2 = tp2[p*OP_POINT:(p+1)*OP_POINT]
					part_combined = np.add(part1, part2)
					spec1, phase1 = stft(part1)
					#print(spec1.shape)
					spec2, phase2 = stft(part2)
					spec_cb, phase_cb = stft(part_combined)

					audio_spectrum1[counter][0] = spec1
					audio_spectrum2[counter][0] = spec2
					audio_spectrum_cb[counter][0] = spec_cb
					audio_phase1[counter] = phase1
					audio_phase2[counter] = phase2
					audio_phase_cb[counter] = phase_cb
					counter = counter + 1

		return audio_spectrum1, audio_spectrum2, audio_spectrum_cb, audio_phase1, audio_phase2, audio_phase_cb, fnls1, fnls2

	def get_data_per_combination(self):	
		#print(self.chosen[self.seq[self.seq_num][0]])
		#print(self.chosen[self.seq[self.seq_num][1]])	
		audio_spectrum1, audio_spectrum2, audio_spectrum_cb, audio_phase1, audio_phase2, audio_phase_cb, fnls1, fnls2 = self.get_audio(self.seq[self.seq_num][0], self.seq[self.seq_num][1])
		image1, image2 = self.get_image(self.seq[self.seq_num][0], self.seq[self.seq_num][1], fnls1, fnls2)

		train_end = False
		if self.seq_num != len(self.seq)-1:
			self.seq_num += 1
		else:
			train_end = True
					
		return image1, image2, audio_spectrum1, audio_spectrum2, audio_spectrum_cb, audio_phase1, audio_phase2, audio_phase_cb, train_end

	def get_data(self):
		self.seq_num = 0
		image_left = [0]*len(self.seq)
		image_right = [0]*len(self.seq)
		audio_spec_left = [0]*len(self.seq)
		audio_spec_right = [0]*len(self.seq)
		audio_spec_combined = [0]*len(self.seq)
		audio_phase_left = [0]*len(self.seq)
		audio_phase_right = [0]*len(self.seq)
		audio_phase_combined = [0]*len(self.seq)
		cnt = 0
		train_end = False

		while not train_end:
			print("No."+str(cnt+1)+" combination...")
			#print(self.seq[cnt][0]+"  "+self.seq[cnt][1])
			if self.val:
				print("##VALIDATION##")
			image_left[cnt], image_right[cnt], audio_spec_left[cnt], audio_spec_right[cnt], audio_spec_combined[cnt], audio_phase_left[cnt], audio_phase_right[cnt], audio_phase_combined[cnt], train_end = self.get_data_per_combination()
			cnt += 1

		all_len = 0
		csl = [0]*len(self.seq)
		for k in range(len(self.seq)):
			head = all_len
			all_len += np.shape(image_left[k])[0]
			trail = all_len
			csl[k] = [head, trail]

		iml = torch.Tensor(all_len, 3, 3, 224, 224)#.cuda()
		imr = torch.Tensor(all_len, 3, 3, 224, 224)#.cuda()
		aspl = torch.Tensor(all_len, 1, 256, 256)#.cuda()
		aspr = torch.Tensor(all_len, 1, 256, 256)#.cuda()
		aspcb = torch.Tensor(all_len, 1, 256, 256)  #.cuda()
		if torch.cuda.is_available():
			iml = iml.cuda()
			imr = imr.cuda()
			aspl = aspl.cuda()
			aspr = aspr.cuda()
			aspcb = aspcb.cuda()
		aphl = np.zeros((all_len, 512, 256))
		aphr = np.zeros((all_len, 512, 256))
		aphcb = np.zeros((all_len, 512, 256))

		for k in range(len(self.seq)):
			iml[csl[k][0]:csl[k][1]] = image_left[k]
			imr[csl[k][0]:csl[k][1]] = image_right[k]
			aspl[csl[k][0]:csl[k][1]] = audio_spec_left[k]
			aspr[csl[k][0]:csl[k][1]] = audio_spec_right[k]
			aspcb[csl[k][0]:csl[k][1]] = audio_spec_combined[k]
			aphl[csl[k][0]:csl[k][1]] = audio_phase_left[k]
			aphr[csl[k][0]:csl[k][1]] = audio_phase_right[k]
			aphcb[csl[k][0]:csl[k][1]] = audio_phase_combined[k]


		shuffle_order = np.arange(all_len)
		np.random.shuffle(shuffle_order)

		iml = iml[shuffle_order]
		imr = imr[shuffle_order]
		aspl = aspl[shuffle_order]
		aspr = aspr[shuffle_order]
		aspcb = aspcb[shuffle_order]
		aphl = aphl[shuffle_order]
		aphr = aphr[shuffle_order]
		aphcb = aphcb[shuffle_order]

		return iml, imr, aspl, aspr, aspcb, aphl, aphr, aphcb
