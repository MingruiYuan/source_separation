#System libs
import os
import random
import math
#Installed libs
import torch
import numpy as np
import torch.optim as optim
#Our libs
from utils import adjust_sound
from myNet import *
from preprocess.SPLdataset import SPLdataset as SPL

#os.environ['CUDA_VISIBLE_DEVICES']='2'
K = 16  # features pre frame
DATA_NUM = 9 # number of npz files
BATCH_SIZE = 64

LOAD_DICT = False # whether load old params
timestr = time.strftime("%d-%H-%M-%S", time.localtime())
STORE_DIR = 'train_1-1_all_' + timestr + '_train_log/' # output dir
SAVE_DATA = True 
CUDA = 1
PARALLEL = False # whether use multiple gpus
if not PARALLEL:
    BATCH_SIZE = 16 # for single gpu 
if not os.path.exists(STORE_DIR):
    os.mkdir(STORE_DIR)

if torch.cuda.is_available() and CUDA:
    mynet = myNet18(pretrained=True, K=K).cuda()
    myU =  U_Net(K=K).cuda()
    mySynthesizer = Audio_Synthesizer(K=K).cuda()
    if PARALLEL:
        mynet = nn.DataParallel(mynet)
        myU = nn.DataParallel(myU)
        mySynthesizer = nn.DataParallel(mySynthesizer)
else:
    myU = U_Net(K=K)
    mynet = myNet18(pretrained=True, K=K)
    mySynthesizer = Audio_Synthesizer(K=K)
if LOAD_DICT:
    myU.load_state_dict(torch.load('train_last_01-12-06-01_train_log/26audio100.t7')['state_dict'])
    mynet.load_state_dict(torch.load('train_last_01-12-06-01_train_log/26video100.t7')['state_dict'])
    mySynthesizer.load_state_dict(torch.load('train_last_01-12-06-01_train_log/26Synthesizer100.t7')['state_dict'])
f = open(STORE_DIR + 'log', 'w')

video_opt = optim.SGD(mynet.parameters(), lr=1e-3,momentum=0.9)
audio_opt = optim.SGD(myU.parameters(), lr=1e-3,momentum=0.9)
syn_opt=optim.SGD(mySynthesizer.parameters(), lr=1e-3,momentum=0.9)
T = np.load('../npz_data/test.npz')
image_left_test = torch.from_numpy(T['image_left']).cuda().float()
image_right_test = torch.from_numpy(T['image_right']).cuda().float()
spec_left_test = torch.from_numpy(T['spec_left']).cuda().float()
spec_right_test = torch.from_numpy(T['spec_right']).cuda().float()
spec_cb_test = torch.from_numpy(T['spec_cb']).cuda().float()
phase_left_test = torch.from_numpy(T['phase_left']).cuda().float()
phase_right_test = torch.from_numpy(T['phase_right']).cuda().float()
phase_cb_test = torch.from_numpy(T['phase_cb']).cuda().float()
for epoch in range(3000):
    avgloss = 0
    avgsdr = 0
    num = 0
    numsdr = 0
    dataSeq = np.arange(DATA_NUM)
    np.random.shuffle(dataSeq)

    # compute sdr for test data
    if epoch % 50 == 49:
        avgsdr = 0
        for j in range(10):
            curImgL, curImgR, curSpecL, curSpecR, curSpecCb, curPhaL, curPhaR, curPhaCb = random_set(image_left_test, image_right_test, spec_left_test, spec_right_test, spec_cb_test, phase_left_test, phase_right_test, phase_cb_test, validate=True, BATCH_SIZE=1)
            sdr, sir, sar = validate(mynet, myU, mySynthesizer, curImgL, curImgR, curSpecL, curSpecR, curSpecCb, curPhaL, curPhaR, curPhaCb, STORE_DIR, CUDA=1)
            avgsdr += sdr[0]
            avgsdr += sdr[1]
        print('epoch: ', str(epoch), 'test', 'sdr: ', avgsdr / 20)
    avgloss = 0
    avgsdr = 0
    num = 0
    numsdr = 0

    # train data
    for i in range(DATA_NUM):
        avgloss = 0
        num = 0

        m = random.randint(0, DATA_NUM - 1)
        D = np.load('../npz_data/'+str(dataSeq[i])+'.npz')
        image_left = torch.from_numpy(D['image_left']).cuda().float()
        image_right = torch.from_numpy(D['image_right']).cuda().float()
        spec_left = torch.from_numpy(D['spec_left']).cuda().float()
        spec_right = torch.from_numpy(D['spec_right']).cuda().float()
        spec_cb = torch.from_numpy(D['spec_cb']).cuda().float()
        phase_left = torch.from_numpy(D['phase_left']).cuda().float()
        phase_right = torch.from_numpy(D['phase_right']).cuda().float()
        phase_cb = torch.from_numpy(D['phase_cb']).cuda().float()

        for j in range(np.floor(image_left.shape[0] / BATCH_SIZE).astype(np.int16)):

            curImgL, curImgR, curSpecL, curSpecR = random_set(image_left, image_right, spec_left, spec_right, BATCH_SIZE=BATCH_SIZE)

            if epoch % 200 == 0 and i == 0 and j == 0 and SAVE_DATA:
                save = STORE_DIR + str(epoch) + '.npz'
                loss, left_loss, right_loss = train(mynet, myU, mySynthesizer, curImgL, curImgR, curSpecL, curSpecR, save_result=save)
            else:
                loss, left_loss, right_loss = train(mynet, myU, mySynthesizer, curImgL, curImgR, curSpecL, curSpecR)
            log = 'epoch: ' + str(epoch) + ' step: ' + str(i) + ' loss:' + str(loss.item()) + 'left_loss:' + str(left_loss) + 'right_loss:' + str(right_loss) + '\n'
            avgloss += loss.item()
            num += 1
            f.writelines(log)
            print(log)

            video_opt.zero_grad()
            audio_opt.zero_grad()
            syn_opt.zero_grad()
            loss.backward()
            video_opt.step()
            audio_opt.step()
            syn_opt.step()
        print('epoch:', epoch, ' avgloss: ', avgloss / num)

    if epoch % 100 == 0:
        torch.save({
              'epoch': epoch,
              'state_dict': mynet.state_dict(), 
            }, STORE_DIR + '1video' + str(epoch) + '.t7')
        torch.save({
              'epoch': epoch,
              'state_dict': myU.state_dict(), 
            }, STORE_DIR + '1audio' + str(epoch) + '.t7')
        torch.save({
              'epoch': epoch,
              'state_dict': mySynthesizer.state_dict(), 
            }, STORE_DIR + '1Synthesizer'+ str(epoch) + '.t7')
        for p in audio_opt.param_groups:
            p['lr'] = p['lr'] * 0.99 ** (epoch // 200)
        for p in video_opt.param_groups:
            p['lr'] = p['lr'] * 0.99 ** (epoch // 200)
        for p in syn_opt.param_groups:
            p['lr'] = p['lr'] * 0.99 ** (epoch // 200)
        print('audio lr:', audio_opt.param_groups[0]['lr'])
        print('video lr:', video_opt.param_groups[0]['lr'])
        print('syn lr:', syn_opt.param_groups[0]['lr'])
