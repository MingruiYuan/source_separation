from myNet import *
from preprocess.SPLdataset import SPLdataset as SPL
from random import randint
import math
import torch
import numpy as np
import os

def init():
    #ROOT_DIR = './'
    ROOT_DIR = '../module/'
    video_net = myNet18(K=16)
    audio_net = U_Net(K=16)
    syn = Audio_Synthesizer(K=16)
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
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    DATA_NUM = 8
    TEST = True # using test or train data
    vn, an, syn = init()
    total_sdr = 0
    total_num = 0
    if TEST:
        STORE_DIR = 'test_result/'
        D = np.load('../npz_data/test.npz')
        image_left = torch.from_numpy(D['image_left']).cuda().float()
        image_right = torch.from_numpy(D['image_right']).cuda().float()
        spec_left = torch.from_numpy(D['spec_left']).cuda().float()
        spec_right = torch.from_numpy(D['spec_right']).cuda().float()
        spec_cb = torch.from_numpy(D['spec_cb']).cuda().float()
        phase_left = torch.from_numpy(D['phase_left']).cuda().float()
        phase_right = torch.from_numpy(D['phase_right']).cuda().float()
        phase_cb = torch.from_numpy(D['phase_cb']).cuda().float()
        cur_sdr = 0
        cur_num = 0
        for j in range(image_left.shape[0]):
            il = image_left[[j]]
            ir = image_right[[j]]
            sl = spec_left[[j]]
            sr = spec_right[[j]]
            scb = spec_cb[[j]]
            pl = phase_left[[j]]
            pr = phase_right[[j]]
            pcb = phase_cb[[j]]
            sdr, sir, sar = validate(vn, an, syn, il, ir, sl, sr, scb, pl, pr, pcb, STORE_DIR, CUDA=1, always=True)
            cur_sdr += sdr[0]
            cur_sdr += sdr[1]
            cur_num += 2
            
        total_sdr += cur_sdr
        total_num += cur_num
        print('cursdr: ', cur_sdr / cur_num)
    else:
        STORE_DIR = 'train_result/'

        for i in range(DATA_NUM):
            D = np.load('../data_preload/'+str(i)+'.npz')
            image_left = torch.from_numpy(D['image_left']).cuda()
            image_right = torch.from_numpy(D['image_right']).cuda()
            spec_left = torch.from_numpy(D['spec_left']).cuda()
            spec_right = torch.from_numpy(D['spec_right']).cuda()
            spec_cb = torch.from_numpy(D['spec_cb']).cuda()
            phase_left = torch.from_numpy(D['phase_left']).cuda()
            phase_right = torch.from_numpy(D['phase_right']).cuda()
            phase_cb = torch.from_numpy(D['phase_cb']).cuda()
            cur_sdr = 0
            cur_num = 0
            for j in range(5):
                il = image_left[[j]]
                ir = image_right[[j]]
                sl = spec_left[[j]]
                sr = spec_right[[j]]
                scb = spec_cb[[j]]
                pl = phase_left[[j]]
                pr = phase_right[[j]]
                pcb = phase_cb[[j]]
                sdr, sir, sar = validate(vn, an, syn, il, ir, sl, sr, scb, pl, pr, pcb, STORE_DIR, CUDA=CUDA, always=True)
                cur_sdr += sdr[0]
                cur_sdr += sdr[1]
                cur_num += 2

            total_sdr += cur_sdr
            total_num += cur_num
            print('cursdr: ', cur_sdr / cur_num)
        print('totalsdr:', total_sdr / total_num)