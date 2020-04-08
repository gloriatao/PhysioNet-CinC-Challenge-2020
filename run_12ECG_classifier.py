#!/usr/bin/env python

import numpy as np
import torch
import cv2
import random
from stft import stft, amplitude_to_db
from resnet_ori import resnet50
import torch.nn as nn

def zero_pad_random_draw(array, length):
    l = array.shape[1]
    if l > length:
        tmp = l - length
        a = random.sample(np.arange(tmp).tolist(), 1)
        a = a[0]
        result = array[:, a:length+a]
    else:
        zero_pad = np.zeros((array.shape[0], length-l))
        result = np.concatenate([array, zero_pad],axis=1)
    return result

def preprocess_ECG(samples):

    samples = np.asfortranarray(samples)
    samples = zero_pad_random_draw(samples, length=3000)

    out = np.zeros((12, 512, 512), dtype=np.float32)
    for i in range(samples.shape[0]):
        sample = np.asfortranarray(samples[i, :])
        spec = stft(sample, n_fft=1024, hop_length=24)
        magnitude = (amplitude_to_db(spec, ref=np.max) + 1.0e-6)[:512]
        magnitude_rescale = np.zeros((512, magnitude.shape[1]), dtype=np.float32)
        ori_row_index = 0
        des_row_index = 0
        while des_row_index < 512:
            if ori_row_index <= 40:
                bin_thickness = 4
            elif ori_row_index <= 80:
                bin_thickness = 4
            elif ori_row_index <= 120:
                bin_thickness = 3
            elif ori_row_index <= 160:
                bin_thickness = 2
            else:
                bin_thickness = 1

            tmp1 = magnitude[ori_row_index, :]
            tmp1 = tmp1[np.newaxis, :]
            tmp1 = np.repeat(tmp1, bin_thickness, axis=0)
            magnitude_rescale[des_row_index:(des_row_index + bin_thickness), :] = tmp1

            des_row_index += bin_thickness
            ori_row_index += 1

        magnitude_rescale = cv2.resize(magnitude_rescale, (512, 512), interpolation=cv2.INTER_NEAREST)
        out[i, :, :] = -magnitude_rescale

    out = out - np.min(out)
    out = out / np.max(out)

    return out

def run_12ECG_classifier(data,header_data,classes,model):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % str(0) if use_cuda else "cpu")
    model.to(device)
    model.eval()

    try:
        age = np.int(header_data[13].split(':')[1])
        age = torch.tensor([age / 100]).float()
    except:
        age = torch.tensor([0]).float()
        print("age nan")

    try:
        sex = str(header_data[14].split(':')[1])
        sex = sex.split('\n')[0]
        if sex[0][0] == 'Male':
            sex = np.ones((1), dtype=np.float32)
        else:
            sex = np.zeros((1), dtype=np.float32)
        sex = torch.tensor(sex).float()
    except:
        sex = torch.tensor([-1]).float()
        print("sex nan")


    # Use your classifier here to obtain a label and score for each class.
    features=preprocess_ECG(data)
    features = torch.tensor(features).float()
    features = features[None,:,:,:]
    age = age[None,:]
    sex = sex[None,:]
    features, sex, age = features.to(device), sex.to(device), age.to(device)
    with torch.no_grad():
        current_score = model.forward(features, sex, age)
        current_score = nn.functional.sigmoid(current_score)

    current_label = current_score.clone()
    current_label[current_label > 0.5] = 1
    current_label[current_label <= 0.5] = 0

    if use_cuda:
        current_label = np.array(current_label.cpu()).astype(np.uint8)
        current_score = np.array(current_score.cpu()).astype(np.float32)
    else:
        current_label = np.array(current_label).astype(np.uint8)
        current_score = np.array(current_score).astype(np.float32)



    return current_label, current_score

def load_weights(model, weightfile):
    params = torch.load(weightfile, map_location=torch.device('cpu'))
    if 'seen' in params.keys():
        model.seen = params['seen']
        del params['seen']
    else:
        model.seen = 0
    model.load_state_dict(params['state_dict'])
    print('Load Weights from %s... Done!!!' % weightfile)
    del params
    return model

def load_12ECG_model():
    # load the model from disk 
    # filename='finalized_model.sav'
    filename = 'resnet50checkpoint.pth'

    loaded_model = resnet50(num_classes=9)
    loaded_model = load_weights(loaded_model, filename)

    return loaded_model
