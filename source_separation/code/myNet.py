#system libs
import math
import time
import os
import random
#Installed libs
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet, model_urls
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
from scipy.io import wavfile
from PIL import Image
from tensorboardX import SummaryWriter

#Our libs
from preprocess.SPLdataset import SPLdataset as SPL
import unet_model
from unet_parts import *
from utils import adjust_sound, recover_sound, compute_validation
from preprocess.STFT import istft

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding and dilation"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=dilation, bias=False)


class myBlock(nn.Module):
    '''
    add dilation param
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(
            inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class myResnet(nn.Module):
    '''
    NN of <The Sound of Pixels>
    '''

    def __init__(self, block, u_channels=1, K=15, base_feature = 16):  # K is the feature per frame
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.final_conv = nn.Conv2d(512, K, kernel_size=3)  # added
        self.spatial_maxpool = nn.AdaptiveMaxPool2d(1)  # added
        self.out = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1, 0.2)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1,x2,x3):

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.final_conv(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.final_conv(x2)

        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)
        x3 = self.layer1(x3)
        x3 = self.layer2(x3)
        x3 = self.layer3(x3)
        x3 = self.layer4(x3)
        x3 = self.final_conv(x3)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x = torch.cat((x1, x2, x3), dim=1)
        x = torch.max(x, dim=1, keepdim=False)[0]

        x = self.spatial_maxpool(x)
        x = self.out(x)
        return x


def myNet18(pretrained=False, K=16, **kwargs):
    """Constructs a model with resnet18 pretrained params.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = myResnet(BasicBlock, K=K, **kwargs)
    if pretrained:
        # strict=false to allow modifications
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet18']), strict=False)
    i = 0
    for name, params in model.named_parameters():
        if i < 60:
            params.requires_grad = False
        i += 1
    return model

class U_Net(nn.Module):
    def __init__(self, u_channels=1, K=16, base_feature = 16):  # K is the feature per frame
        super().__init__()
        # u-net implementation

        self.down1 = double_conv(u_channels, base_feature)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(base_feature, base_feature*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = double_conv(base_feature*2, base_feature*4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = double_conv(base_feature*4, base_feature*8)
        self.pool4 = nn.MaxPool2d(2)
        self.down5 = double_conv(base_feature * 8, base_feature * 16)
        self.pool5 = nn.MaxPool2d(2)
        self.down6 = double_conv(base_feature * 16, base_feature * 32)
        self.pool6 = nn.MaxPool2d(2)
        self.bottom = double_conv(base_feature * 32, base_feature * 64)
        self.up6 = nn.ConvTranspose2d(base_feature * 64, base_feature * 32, 2, 2)
        self.up_conv6 = double_conv(base_feature * 64, base_feature * 32)
        self.up5 = nn.ConvTranspose2d(base_feature * 32, base_feature * 16, 2, 2)
        self.up_conv5 = double_conv(base_feature * 32, base_feature * 16)
        self.up4 = nn.ConvTranspose2d(base_feature * 16, base_feature * 8, 2, 2)
        self.up_conv4 = double_conv(base_feature * 16, base_feature * 8)
        self.up3 = nn.ConvTranspose2d(base_feature * 8, base_feature * 4, 2, 2)
        self.up_conv3 = double_conv(base_feature * 8, base_feature * 4)
        self.up2 = nn.ConvTranspose2d(base_feature * 4, base_feature * 2, 2, 2)
        self.up_conv2 = double_conv(base_feature * 4, base_feature * 2)
        self.up1 = nn.ConvTranspose2d(base_feature * 2, base_feature, 2, 2)
        self.up_conv1 = double_conv(base_feature * 2, base_feature)
        self.u_outc = outconv(base_feature, K)
        self.act = nn.ReLU() # abandoned
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, y):

        y1 = self.down1(y)
        pool1 = self.pool1(y1)
        
        y2 = self.down2(pool1)
        pool2 = self.pool2(y2)
        
        y3 = self.down3(pool2)
        pool3 = self.pool3(y3)
        
        y4 = self.down4(pool3)
        pool4 = self.pool4(y4)
        
        y5 = self.down5(pool4)
        pool5 = self.pool5(y5)

        y6 = self.down6(pool5)
        pool6 = self.pool6(y6)

        y7 = self.bottom(pool6)

        y_up6 = self.up6(y7)
        y_cat6 = torch.cat([y6, y_up6], dim=1)
        y_up_conv6 = self.up_conv6(y_cat6)

        y_up5 = self.up5(y_up_conv6)
        y_cat5 = torch.cat([y5, y_up5], dim=1)
        y_up_conv5 = self.up_conv5(y_cat5)
        
        y_up4 = self.up4(y_up_conv5)
        y_cat4 = torch.cat([y4, y_up4], dim=1)
        y_up_conv4 = self.up_conv4(y_cat4)

        y_up3 = self.up3(y_up_conv4)
        y_cat3 = torch.cat([y3, y_up3], dim=1)
        y_up_conv3 = self.up_conv3(y_cat3)

        y_up2 = self.up2(y_up_conv3)
        y_cat2 = torch.cat([y2, y_up2], dim=1)
        y_up_conv2 = self.up_conv2(y_cat2)

        y_up1 = self.up1(y_up_conv2)
        y_cat1 = torch.cat([y1, y_up1], dim=1)
        y_up_conv1 = self.up_conv1(y_cat1)
        y = self.u_outc(y_up_conv1)

        return y

class Audio_Synthesizer(nn.Module):
    def __init__(self, K=16):
        super().__init__()
        self.linear = nn.Linear(K, 1)
        self.sigmoid = nn.Sigmoid()
        self.linear.weight.data.normal_(0, 0.0001)
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def random_set(allImgL, allImgR, allSpecL, allSpecR, allSpecCb = None, allPhaL=None, allPhaR = None, allPhaCb=None, validate=False, BATCH_SIZE=1):
    '''
    allImgL: all,3,3,224,224
    allSpecL: all,1,256,256
    allPhaL: all,512,256

    '''
    listSize = allImgL.shape[0]
    
    if validate:
        seek = np.random.randint(0, listSize, BATCH_SIZE)

        if torch.is_tensor(allPhaL) and torch.is_tensor(allPhaR) and torch.is_tensor(allPhaCb):
            return allImgL[seek,:,:,:,:], allImgR[seek,:,:,:,:], allSpecL[seek,:,:,:], allSpecR[seek,:,:,:], allSpecCb[seek,:,:,:], allPhaL[seek,:,:].squeeze(0), allPhaR[seek,:,:].squeeze(0), allPhaCb[seek,:,:].squeeze(0)
        else:
            print('ERROR: need phases of spec to validate')
            raise ValueError
            return None, None, None, None, None, None, None, None
    else:
        # shuffle the combination
        seekl = np.random.randint(0, listSize, BATCH_SIZE)
        seekr = np.random.randint(0, listSize, BATCH_SIZE)
        random.shuffle(seekl)
        random.shuffle(seekr)
        return allImgL[seekl,:,:,:,:], allImgR[seekr,:,:,:,:], allSpecL[seekl,:,:,:], allSpecR[seekr,:,:,:]
        
def gtGenerater(spec_left, spec_right, CUDA=1):
    '''
    spec_left, spec_right: b,1,256,256
    return: 2,b,1,256,256
    '''

    spec_array = torch.empty(2,*spec_left.shape) # 2,b,1,256,256
    spec_array[0,:,:,:,:] = spec_left
    spec_array[1,:,:,:,:] = spec_right
    gt_mask = torch.argmax(spec_array, dim=0)
    mask_gt = torch.empty(*spec_array.shape)
    if torch.cuda.is_available() and CUDA:
        mask_gt = mask_gt.cuda()
    mask_gt[0,:,:,:,:] = (gt_mask == 0).type(torch.float32)
    mask_gt[1,:,:,:,:] = (gt_mask == 1).type(torch.float32)
    return mask_gt
def train(videoNet, audioNet, synNet,image_left, image_right, spec_left, spec_right, validate = False, save_result=None, CUDA=1, test = False, spec_cb = None):
    """
    image_left: b,3,3,224,224
    image_right: b,3,3,224,224
    spec_left: b,1,256,256
    spec_right: b,1,256,256
    save_result:
        'left_image':m,K
        'audio': m, K,256,256
        'mask_left': m,256,256
    return: loss, sepration 2,256,256
    """
    if not validate:
        switch = random.randint(0, 1)
        if switch:
            temp = image_left
            image_left = image_right
            image_right = temp
            temp = spec_left
            spec_left = spec_right
            spec_right = temp
            print('left-right-switched')
    mask_gt_array = gtGenerater(spec_left, spec_right, CUDA=CUDA) # 2,b,1,256,256
    if not test:
        spec_cb = spec_left + spec_right
    else:
        pass
    if torch.cuda.is_available() and CUDA:
        image_left = image_left.cuda()
        image_right = image_right.cuda()

    total_loss = None
    audio_out=audioNet(spec_cb) # b,K,256,256
    sepration = torch.zeros(2, *spec_left.shape)  # 2,b,1,256,256

    if torch.cuda.is_available() and CUDA:
        sepration = sepration.cuda()
    # left
    video_out_left = videoNet(image_left[:,0,:,:,:], image_left[:,1,:,:,:], image_left[:,2,:,:,:]) # b,K,1,1
    mask_left = video_out_left * audio_out
    mask_left = torch.transpose(mask_left, 1, 2)
    mask_left = torch.transpose(mask_left, 2, 3)  # (b, 256, 256, K)
    mask_left_linear = synNet(mask_left) # b,256,256,1
    mask_left_linear = torch.transpose(mask_left_linear, 2, 3)
    mask_left_linear = torch.transpose(mask_left_linear, 1, 2)  # b,1,256,256
    mask_gt = mask_gt_array[0] # b,1,256,256
    if torch.cuda.is_available() and CUDA:
        mask_gt = mask_gt.cuda()
    total_loss = nn.functional.binary_cross_entropy(mask_left_linear, mask_gt)
    left_loss = total_loss.item()
    mask_left_linear_detach = mask_left_linear.clone().detach()
    mask_left_linear_detach[mask_left_linear_detach > 0.9] = 1
    mask_left_linear_detach[mask_left_linear_detach < 0.1] = 0
    sepration[0,:,:,:,:] = mask_left_linear_detach
    # right
    video_out_right = videoNet(image_right[:,0,:,:,:], image_right[:,1,:,:,:], image_right[:,2,:,:,:]) # b,K,1,1
    mask_right = video_out_right * audio_out
    mask_right = torch.transpose(mask_right, 1, 2)
    mask_right = torch.transpose(mask_right, 2, 3)  # (b, 256, 256, K)
    mask_right_linear = synNet(mask_right) # b,256,256,1
    mask_right_linear = torch.transpose(mask_right_linear, 2, 3)
    mask_right_linear = torch.transpose(mask_right_linear, 1, 2)  # b,1,256,256
    mask_gt = mask_gt_array[1] # b,1,256,256
    if torch.cuda.is_available() and CUDA:
        mask_gt = mask_gt.cuda()
    total_loss += nn.functional.binary_cross_entropy(mask_right_linear, mask_gt)
    right_loss = total_loss.item() - left_loss
    mask_right_linear_detach = mask_right_linear.clone().detach()
    mask_right_linear_detach[mask_right_linear_detach > 0.9] = 1
    mask_right_linear_detach[mask_right_linear_detach < 0.1] = 0
    
    sepration[1,:,:,:,:] = mask_right_linear_detach
    if save_result is not None:
        np.savez(save_result, vol = video_out_left.detach().cpu().numpy(),vor = video_out_right.detach().cpu().numpy(), ao = audio_out.detach().cpu().numpy(), mlsep = mask_left_linear.detach().cpu().numpy(), mrsep = mask_right_linear.detach().cpu().numpy(), maskgt=mask_gt_array.detach().cpu().numpy(), mask_left = mask_left.detach().cpu().numpy(), mask_right = mask_right.detach().cpu().numpy())

    if validate:
        return [total_loss, left_loss, right_loss, sepration.squeeze().detach().cpu()] # 2,256,256
    else:
        return [total_loss, left_loss, right_loss]



def validate(videoNet, audioNet, synNet, image_left, image_right, spec_left, spec_right, spec_cb, phase_left, phase_right, phase_cb, STORE_DIR, CUDA=1, always=False, test = False):
    """
    validate sdr and save seprated files
    image_left: 1,3,3,224,224
    spec_left:1,1,256,256
    spec_cb: 1,1,256,256
    phase_left : 1,256,256
    """
    STORE_WAV = 'validate_wav/'
    if not os.path.exists(STORE_DIR):
        os.makedirs(STORE_DIR)
    if not os.path.exists(STORE_DIR + STORE_WAV):
        os.makedirs(STORE_DIR + STORE_WAV)
    timestr = time.strftime("%d-%H-%M-%S", time.localtime())
    renormalize = transforms.Compose(
        [transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])]
        )
    if not always:
        videoNet.eval()
        audioNet.eval()
        synNet.eval()
    if not test:
        _,left_loss, right_loss,  sepration = train(videoNet, audioNet, synNet, image_left, image_right, spec_left, spec_right, validate=True, CUDA=CUDA)
    else:
        _,left_loss, right_loss,  sepration = train(videoNet, audioNet, synNet, image_left, image_right, spec_left, spec_right, validate=True, CUDA=CUDA, test=1, spec_cb=spec_cb)
    sepration = sepration.cuda()

    if not always:
        videoNet.train()
        audioNet.train()
        synNet.train()

    mask_gt = gtGenerater(spec_left, spec_right, CUDA=CUDA).squeeze().cpu().numpy()  # 2,256,256
    sepration = sepration.cpu().numpy()
    spec_right = spec_right.squeeze().cpu().numpy() # 256,256
    spec_left = spec_left.squeeze().cpu().numpy()
    spec_cb = spec_cb.squeeze().cpu().numpy()
    phase_left = phase_left.squeeze().cpu().numpy() # 512,256
    phase_right = phase_right.squeeze().cpu().numpy()
    phase_cb = phase_cb.squeeze().cpu().numpy()

    target_spec_left = mask_gt[0] * spec_cb
    target_spec_right = mask_gt[1] * spec_cb
    target_left = istft(target_spec_left, phase_cb)
    target_right = istft(target_spec_right, phase_cb)

    sep_spec_left = sepration[0] * spec_cb
    sep_spec_right = sepration[1] * spec_cb
    sep_left = istft(sep_spec_left, phase_cb)
    sep_right = istft(sep_spec_right, phase_cb)

    diff_spec_left = np.abs(mask_gt[0] - sepration[0]) * spec_cb
    diff_spec_right = np.abs(mask_gt[1] - sepration[1]) * spec_cb
    diff_left = istft(diff_spec_left, phase_cb)
    diff_right = istft(diff_spec_right, phase_cb)

    mr = np.mean(np.abs(mask_gt[1] - sepration[1]))
    ref_left = istft(spec_left, phase_left)
    ref_right = istft(spec_right, phase_right)
    ref_cb = istft(spec_cb, phase_cb)

    
    image_left = image_left.squeeze()  # 3,3,224,224
    image_right = image_right.squeeze()

    for i in range(3):
        img_left = transforms.functional.to_pil_image(renormalize(image_left[i,:,:,:]).cpu())
        img_right = transforms.functional.to_pil_image(renormalize(image_right[i,:,:,:]).cpu())
        img_left.save(STORE_DIR + STORE_WAV + timestr + 'left' + str(i) + '.jpg')
        img_right.save(STORE_DIR + STORE_WAV + timestr + 'right' + str(i) + '.jpg')
    
    wavfile.write(STORE_DIR + STORE_WAV + timestr + '_sep_left_.wav', 11025, sep_left.astype(np.int16))
    #wavfile.write(STORE_DIR + STORE_WAV + timestr + '_target_left_.wav', 11025, target_left.astype(np.int16))
    wavfile.write(STORE_DIR + STORE_WAV + timestr + '_gt_left_.wav', 11025, ref_left.astype(np.int16))
    wavfile.write(STORE_DIR + STORE_WAV + timestr + '_sep_right_.wav', 11025, sep_right.astype(np.int16))
    #wavfile.write(STORE_DIR + STORE_WAV + timestr + '_target_right_.wav', 11025, target_right.astype(np.int16))
    wavfile.write(STORE_DIR + STORE_WAV + timestr + '_gt_right_.wav', 11025, ref_right.astype(np.int16))
    wavfile.write(STORE_DIR + STORE_WAV + timestr + '_gt_cb_.wav', 11025, ref_cb.astype(np.int16))
    #wavfile.write(STORE_DIR + STORE_WAV + timestr + '_diff_left.wav', 11025, diff_left.astype(np.int16))
    #wavfile.write(STORE_DIR + STORE_WAV + timestr + '_diff_right.wav', 11025, diff_right.astype(np.int16))
    
    np.savez(STORE_DIR + STORE_WAV + timestr + 'mask.npz', ml=sepration[0], mlgt=mask_gt[0], mr=sepration[1], mrgt=mask_gt[1], left_loss=left_loss, right_loss=right_loss)
    np.savez(STORE_DIR + STORE_WAV + timestr + 'data.npz', image_left=image_left, image_right=image_right, spec_cb=spec_cb, spec_left=spec_left, spec_right=spec_right)
    refs = np.concatenate((ref_left[np.newaxis,:], ref_right[np.newaxis,:]))
    seps = np.concatenate((sep_left[np.newaxis,:], sep_right[np.newaxis,:]))
    
    sdr, sir, sar = compute_validation(refs, seps)
    print('sdr:', (sdr[0]+ sdr[1]) / 2, 'name: ', timestr)
    return [sdr, sir, sar]

if __name__ == "__main__": # for test
    CUDA=1
    K = 16  # features pre frame
    BATCH_SIZE = 3
    STORE_DIR = 'test-mynet/'
    if torch.cuda.is_available() and CUDA:
        mynet = myNet18(pretrained=True, K=K).cuda()
        myU = U_Net(K=K).cuda()
        mySynthesizer = Audio_Synthesizer(K=K).cuda()
    else:
        myU = U_Net(K=K)
        mynet = myNet18(pretrained=True, K=K)
        mySynthesizer = Audio_Synthesizer(K=K)


    x = torch.rand(BATCH_SIZE, 3, 3, 224, 224) # test input frame epoch x 3 x channel x H x W
    y = torch.rand(BATCH_SIZE, 1, 256, 256) # test input audio epoch x channel x T x F
    i = 0
    writer = SummaryWriter(log_dir='audio_net')
    with writer:
        writer.add_graph(myU, (y, ))
    a = mynet(x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]) # video output 
    b = myU(y)  # audio output
    z = a * b # epoch x channel x T x F
    z = torch.transpose(z, 1, 2)
    z = torch.transpose(z, 2, 3) # epoch x T x F x channel
    z = mySynthesizer(z) # epoch x T x F x 1
    z = torch.transpose(z, 3, 2)
    z = torch.transpose(z, 2, 1) # epoch x 1 x T x F

