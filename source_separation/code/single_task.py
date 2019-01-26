from myNet import *
from preprocess.SPLdataset import SPLdataset as SPL
from random import randint
import math
import torch
import numpy as np
import os

def init():
    #ROOT_DIR = './'
    ROOT_DIR = '../module/single_task/'
    video_net = myNet18(K=10)
    audio_net = U_Net(K=10)
    syn = Audio_Synthesizer(K=10)
    video_net = video_net.cuda()
    audio_net = audio_net.cuda()
    syn = syn.cuda()
    video_net.load_state_dict(torch.load(ROOT_DIR + 'video.t7')['state_dict'])
    audio_net.load_state_dict(torch.load(ROOT_DIR + 'audio.t7')['state_dict'])
    syn.load_state_dict(torch.load(ROOT_DIR + 'Synthesizer.t7')['state_dict'])

    video_net.eval()
    audio_net.eval()
    syn.eval()

    return video_net, audio_net, syn
    


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES']='2'

    STORE_DIR = 'single_task/'
    CUDA=1
    vn, an, syn = init()

            
    spl = SPL([0], [1,2])
    image_left, image_right, spec_left, spec_right, spec_cb, phase_left, phase_right, phase_cb = spl.get_data()    
    image_left = image_left.cuda()
    image_right = image_right.cuda()
    spec_right = spec_right.cuda()
    spec_cb = spec_cb.cuda()
    phase_left = torch.from_numpy(phase_left).cuda()

    phase_right = torch.from_numpy(phase_right).cuda()
    phase_cb = torch.from_numpy(phase_cb).cuda()
    for i in range(image_left.shape[0]):
        il = image_left[[i]]
        ir = image_right[[i]]
        sl = spec_left[[i]]
        sr = spec_right[[i]]
        scb = spec_cb[[i]]
        pl = phase_left[[i]]
        pr = phase_right[[i]]
        pcb = phase_cb[[i]]
        sdr, sir, sar = validate(vn, an, syn, il, ir, sl, sr, scb, pl, pr, pcb, STORE_DIR, CUDA=CUDA, always=True)
        print(i, sdr)