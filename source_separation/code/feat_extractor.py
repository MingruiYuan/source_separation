import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

global net
global normalize
global preprocess
global features_blobs
global classes
global weight_softmax
labels_path = 'labels.json'
idxs=[401,402,486,513,558,642,776,889]
names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    global features_blobs
    features_blobs=output.data.cpu().numpy()

def load_model():
    global net
    global normalize
    global preprocess
    global features_blobs
    global classes
    global weight_softmax
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       normalize
    ])
    classes = {int(key):value for (key, value)
              in list(json.load(open(labels_path,'r')).items())}
    if torch.cuda.is_available():
        net=net.cuda()

def get_CAM(imdir,savedir,imname,CAM=False,ins=None,ins_location=None,out=True):
    img_pil = Image.open(os.path.join(imdir,imname))
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    img = cv2.imread(os.path.join(imdir,imname))
    height, width, _ = img.shape
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x=h_x.cpu()
    probs1 = h_x.numpy()
    probs=[]
    #if CAM:
    #    ins_location[ins[0]] = []
    #    ins_location[ins[1]] = []
    for i in range(0, 8):
        #print(('{:.3f} -> {}'.format(probs1[idxs[i]], names[i])))
        if CAM and (ins[0]==names[i] or ins[1]==names[i]):
            CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])        
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
	    #print(heatmap.shape)
            #H, W = heatmap.shape
            mxm = np.max(heatmap[:,:,2])
            loc = np.where(heatmap[:,:,2]==mxm)
            #print(len(loc[0]))
            if names[i] == ins[0]:
                ins_location[ins[0]+'_x'].append(np.mean(loc[0]))
                ins_location[ins[0]+'_y'].append(np.mean(loc[1]))
            if names[i] == ins[1]:
                ins_location[ins[1]+'_x'].append(np.mean(loc[0]))
                ins_location[ins[1]+'_y'].append(np.mean(loc[1]))
            result = heatmap * 0.3 + img * 0.5
            if out:
                cv2.imwrite(savedir+'/'+names[i]+'_'+imname, result)
        
        probs.append(probs1[idxs[i]])
    return probs

def extract(called=None):
    imdir='../testset7/testimage/' # modify this to suit your locale
    imdir += called
    load_model()
    imlist=os.listdir(imdir)
    imlist = imlist[20:120]
    ins_location={}
    probs=np.zeros([8])
    for im in imlist:
        probs1=get_CAM(imdir,'CAM',im)
        probs=probs+np.array(probs1)
    print(probs)
    print(names)
    index = np.argpartition(probs, -2)[-2:]
    ins = [names[index[0]], names[index[1]]]
    ins_location[ins[0]+'_x']=[]
    ins_location[ins[0]+'_y']=[]
    ins_location[ins[1]+'_x']=[]
    ins_location[ins[1]+'_y']=[]
    for im in imlist:
        get_CAM(imdir,'CAM',im,True,ins,ins_location)
    return probs, names, ins_location

if __name__=='__main__':
    extract()
            

