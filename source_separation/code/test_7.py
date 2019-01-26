import os
import numpy as np 
# import cv2
from preprocess.STFT import *
from myNet import *
from scipy.io import wavfile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import scipy.ndimage as sn
from PIL import Image
import time
import json
# from utils_noadjust import init
from feat_extractor import *
OP_POINT = 65280
imgnum = 14.2106
test_dir = '../testset7/'
#STORE_DIR = '/home/zhangwq01/YuanMingrui/server/test_with_val_2/'

# Load Trained Module
def init(ROOT_DIR = '../module/', DICT = '500.t7', CUDA=1):
    #ROOT_DIR = './'
    #ROOT_DIR = 'train_1-1_all_01-18-05-52_train_log/'
    #DICT = '500.t7'
    video_net = myNet18(K=16)
    audio_net = U_Net(K=16)
    syn = Audio_Synthesizer(K=16)
    if torch.cuda.is_available() and CUDA:
        video_net = video_net.cuda()
        audio_net = audio_net.cuda()
        syn = syn.cuda()
    video_net.load_state_dict(torch.load(ROOT_DIR + 'video.t7')['state_dict'])
    audio_net.load_state_dict(torch.load(ROOT_DIR + 'audio.t7')['state_dict'])
    syn.load_state_dict(torch.load(ROOT_DIR + 'Synthesizer.t7')['state_dict'])

    video_net = nn.DataParallel(video_net)
    audio_net = nn.DataParallel(audio_net)
    syn = nn.DataParallel(syn)
    video_net.eval()
    audio_net.eval()
    syn.eval()

    return video_net, audio_net, syn

def SPLtest(image1, audio, output_dir, files, y1, y2):
	CUDA = 1
	imageNet, audioNet, synNet = init(CUDA=CUDA)

	# Preprocess audio(combined audio)
	audio = sn.zoom(audio.astype('float64'), 0.25, order=2)
	audio_length = len(audio)
	curr_N = len(audio) // OP_POINT
	audio_spec = torch.Tensor(curr_N+1, 1, 256, 256)
	if torch.cuda.is_available() and CUDA:
		audio_spec = audio_spec.cuda()
	audio_phase = np.zeros((curr_N+1, 512, 256))
	cnt = 0
	#print('curr_N:', curr_N)
	for p in range(curr_N+1):
		if p == curr_N:
			part = audio[p*OP_POINT:]
			L = len(part)
			part = np.append(part, np.zeros((OP_POINT-L,)))
			part = part.astype('float32')
		else:
			part = audio[p*OP_POINT:(p+1)*OP_POINT]
		spec, phase = stft(part)
		audio_spec[cnt][0] = spec
		audio_phase[cnt] = phase
		cnt += 1

	# Preprocess image
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	normalize = transforms.Compose(
	[transforms.Resize(256),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
	])
	image_left = torch.Tensor(curr_N+1, 3, 3, 224, 224)
	image_right = torch.Tensor(curr_N+1, 3, 3, 224, 224)
	if torch.cuda.is_available() and CUDA:
		image_left = image_left.cuda()
		image_right = image_right.cuda()

	cnt = 0
	seg = int((y1+y2)//2)
	#print(seg)
	for p in range(curr_N+1):
		#print(cnt)
		if p == curr_N:
			x1 = round(p*imgnum)+1
			x3 = min(round((p+1)*imgnum)-1,curr_N-1)
			x2 = (x1+x3) // 2
			if x1 < np.shape(image)[0]:
				im_L1 = image[x1][:,:seg,:]
				im_R1 = image[x1][:,seg:,:]
				im_L1 = Image.fromarray(im_L1.astype('uint8'))
				im_R1 = Image.fromarray(im_R1.astype('uint8'))
				image_left[cnt][0] = normalize(im_L1)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x1]),(224,224))/255, (2,0,1))))
				image_right[cnt][0] = normalize(im_R1)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x2]),(224,224))/255, (2,0,1))))
			else:
				im_L1 = np.zeros((3,224,224))
				im_R1 = np.zeros((3,224,224))
				image_left[cnt][0] = torch.from_numpy(im_L1)
				image_right[cnt][0] = torch.from_numpy(im_R1)
				
			if x2 < np.shape(image)[0]:
				im_L2 = image[x2][:,:seg,:]
				im_R2 = image[x2][:,seg:,:]
				im_L2 = Image.fromarray(im_L2.astype('uint8'))
				im_R2 = Image.fromarray(im_R2.astype('uint8'))
				image_left[cnt][1] = normalize(im_L2)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x2]),(224,224))/255, (2,0,1))))
				image_right[cnt][1] = normalize(im_R2)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x2]),(224,224))/255, (2,0,1))))
			else:
				im_L2 = np.zeros((3,224,224))
				im_R2 = np.zeros((3,224,224))
				image_left[cnt][1] = torch.from_numpy(im_L2)
				image_right[cnt][1] = torch.from_numpy(im_R2)

			if x3 < np.shape(image)[0]:
				im_L3 = image[x3][:,:seg,:]
				im_R3 = image[x3][:,seg:,:]
				im_L3 = Image.fromarray(im_L3.astype('uint8'))
				im_R3 = Image.fromarray(im_R3.astype('uint8'))
				image_left[cnt][2] = normalize(im_L3)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x3]),(224,224))/255, (2,0,1))))
				image_right[cnt][2] = normalize(im_R3)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x3]),(224,224))/255, (2,0,1))))
			else: 
				im_L3 = np.zeros((3,224,224))
				im_R3 = np.zeros((3,224,224))	
				image_left[cnt][2] = torch.from_numpy(im_L3)
				image_right[cnt][2] = torch.from_numpy(im_R3)

		else:
			# save_dir = '/home/zhangwq01/YuanMingrui/server/wwwimage/'
			x1 = round(p*imgnum)+1
			x3 = min(round((p+1)*imgnum)-1, curr_N-1)
			x2 = (x1+x3) // 2
			im_L1 = image[x1][:,:seg,:]
			im_R1 = image[x1][:,seg:,:]
			# cv2.imwrite(save_dir+str(cnt)+'_'+str(x1)+'_left.jpg',im_L1)
			# cv2.imwrite(save_dir+str(cnt)+'_'+str(x1)+'_right.jpg',im_R1)
			im_L2 = image[x2][:,:seg,:]
			im_R2 = image[x2][:,seg:,:]			# cv2.imwrite(save_dir+str(cnt)+'_'+str(x2)+'_left.jpg',im_L2)
			# cv2.imwrite(save_dir+str(cnt)+'_'+str(x2)+'_right.jpg',im_R2)
			im_L3 = image[x3][:,:seg,:]
			im_R3 = image[x3][:,seg:,:]
			# cv2.imwrite(save_dir+str(cnt)+'_'+str(x3)+'_left.jpg',im_L3)
			# cv2.imwrite(save_dir+str(cnt)+'_'+str(x3)+'_right.jpg',im_R3)
			
			im_L1 = Image.fromarray(im_L1.astype('uint8'))
			im_R1 = Image.fromarray(im_R1.astype('uint8'))
			im_L2 = Image.fromarray(im_L2.astype('uint8'))
			im_R2 = Image.fromarray(im_R2.astype('uint8'))
			im_L3 = Image.fromarray(im_L3.astype('uint8'))
			im_R3 = Image.fromarray(im_R3.astype('uint8'))

			image_left[cnt][0] = normalize(im_L1)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x1]),(224,224))/255, (2,0,1))))
				
			image_left[cnt][1] = normalize(im_L2)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x2]),(224,224))/255, (2,0,1))))
				
			image_left[cnt][2] = normalize(im_L3)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile1 + '/' + all_img1[x3]),(224,224))/255, (2,0,1))))
				
			image_right[cnt][0] = normalize(im_R1)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile2 + '/' + all_img2[x1]),(224,224))/255, (2,0,1))))
				
			image_right[cnt][1] = normalize(im_R2)  #normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile2 + '/' + all_img2[x2]),(224,224))/255, (2,0,1))))
				
			image_right[cnt][2] = normalize(im_R3)  # normalize(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(imagefile2 + '/' + all_img2[x3]),(224,224))/255, (2,0,1))))
			cnt += 1		

	# Source Separation
	audio_out = audioNet(audio_spec)
	separation = torch.zeros(2, *audio_spec.shape)
	if torch.cuda.is_available() and CUDA:
		separation = separation.cuda()

	img_out_left = imageNet(image_left[:,0,:,:,:], image_left[:,1,:,:,:], image_left[:,2,:,:,:])
	mask_left = img_out_left*audio_out
	mask_left = torch.transpose(mask_left, 1, 2)
	mask_left = torch.transpose(mask_left, 2, 3)
	mask_left_linear = synNet(mask_left)
	mask_left_linear = torch.transpose(mask_left_linear, 2, 3)
	mask_left_linear = torch.transpose(mask_left_linear, 1, 2)
	mask_left_linear_detach = mask_left_linear.clone().detach()
	mask_left_linear_detach[mask_left_linear_detach > 0.9] = 1
	mask_left_linear_detach[mask_left_linear_detach < 0.1] = 0
	separation[0,:,:,:,:] = mask_left_linear_detach

	img_out_right = imageNet(image_right[:,0,:,:,:], image_right[:,2,:,:,:], image_right[:,2,:,:,:])
	mask_right = img_out_right*audio_out
	mask_right = torch.transpose(mask_right, 1, 2)
	mask_right = torch.transpose(mask_right, 2, 3)
	mask_right_linear = synNet(mask_right)
	mask_right_linear = torch.transpose(mask_right_linear, 2, 3)
	mask_right_linear = torch.transpose(mask_right_linear, 1, 2)
	mask_right_linear_detach = mask_right_linear.clone().detach()
	mask_right_linear_detach[mask_right_linear_detach > 0.9] = 1
	mask_right_linear_detach[mask_right_linear_detach < 0.1] = 0
	separation[1,:,:,:,:] = mask_right_linear_detach

	separation = separation.squeeze().detach().cpu().numpy() # 2*batch*256*256
	audio_spec_sq = audio_spec.squeeze().cpu().numpy() # batch*256*256
	sep_spec_left = separation[0]*audio_spec_sq
	sep_spec_right = separation[1]*audio_spec_sq

	audio_proc_left = np.zeros((OP_POINT*(curr_N+1),))
	audio_proc_right = np.zeros((OP_POINT*(curr_N+1),))
	for i in range(curr_N):
		#print('Eps.'+str(i+1))
		audio_proc_left[i*OP_POINT:(i+1)*OP_POINT] = istft(sep_spec_left[i], audio_phase[i])
		audio_proc_right[i*OP_POINT:(i+1)*OP_POINT] = istft(sep_spec_right[i], audio_phase[i])

	audio_proc_left = audio_proc_left[:audio_length]
	audio_proc_right = audio_proc_right[:audio_length]
	# sdr_left = compute_validation(audio_gt1, audio_proc_left)
	# sdr_right = compute_validation(audio_gt2, audio_proc_right)
	# print('SDR left:', sdr_left[0])
	# print('SDR right:', sdr_right[0])
	wavfile.write(output_dir+files+'_seg1.wav', 11025, audio_proc_left.astype('int16'))
	wavfile.write(output_dir+files+'_seg2.wav', 11025, audio_proc_right.astype('int16'))

#if __name__ == 'main':
start2 = time.time()
filename = os.listdir(test_dir+'testimage/')
output_dir = '../separated_audio/test_7/'
# gt_dir = test_dir+'gt_sep_audio/'
outdict = {}
cnt=0
for files in filename:
	#if cnt==0:
	print(str(cnt+1)+' '+files)
	outdict[files+'.wav'] = []
	fs, audio = wavfile.read(test_dir+'gt_audio/'+files+'.wav')
	probs, names, location = extract(files+'/')
	index = np.argpartition(probs, -2)[-2:]
	#print(index)
	ins0 = names[index[0]]
	ins1 = names[index[1]]
	y1 = np.mean(location[ins0+'_y'])
	y2 = np.mean(location[ins1+'_y'])
	if y1<y2:
		tp1={"audio":files+'_seg1.wav', "position":0}
		tp2={'audio':files+'_seg2.wav', "position":1}
		outdict[files+'.wav']=[tp1,tp2]
	else:
		tp1={"audio":files+'_seg1.wav', "position":1}
		tp2={"audio":files+'_seg2.wav', "position":0}
		outdict[files+'.wav']=[tp1,tp2]
	print(ins0+'_x:',round(np.mean(location[ins0+'_x'])))
	print(ins0+'_y:',round(np.mean(location[ins0+'_y'])))
	print(ins1+'_x:',round(np.mean(location[ins1+'_x'])))
	print(ins1+'_y:',round(np.mean(location[ins1+'_y'])))
	#image1 = Image.open(test_dir+'mono_image/'+ins0+'.jpg')
	#image2 = Image.open(test_dir+'mono_image/'+ins1+'.jpg')
	imagename = os.listdir(test_dir+'testimage/'+files)
	imagename.sort(key = lambda x:int(x[:-4]))
	getsize_img = cv2.imread(test_dir+'testimage/'+files+'/'+imagename[0])
	
	image = np.zeros((len(imagename), *(getsize_img.shape)))
	for k in range(len(imagename)):
		image[k] = cv2.cvtColor(cv2.imread(test_dir+'testimage/'+files+'/'+imagename[k]), cv2.COLOR_BGR2RGB)
	print('Eg.'+str(cnt+1)+'.'+files)
	
	SPLtest(image, audio, output_dir, files, y1, y2)
	cnt += 1
end2 = time.time()
print('time elapsed:', end2-start2)
with open('../json/sep_7.json','w') as f:
	json.dump(outdict, f, sort_keys=False, indent=4, separators=(',',':'))
