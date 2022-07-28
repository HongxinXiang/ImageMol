import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.exceptions import NoSuchNameError, NoIndexError


def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda


def load_image(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    return img


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if cuda_available():
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save(mask, img, gradcam_path, thresh=0):
    mask = (mask - np.min(mask)) / np.max(mask)
    img = img.reshape(224, 224, 3)
    new_mask = mask.copy()
    new_mask[new_mask < thresh] = 0
    heatmap = cv2.applyColorMap(np.uint8(255 * new_mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    gradcam = 1.0 * heatmap + img
    gradcam = gradcam / np.max(gradcam)
    try:
        assert cv2.imwrite(gradcam_path, np.uint8(255 * gradcam))
    except:
        assert cv2.imwrite(gradcam_path, np.uint8(255 * gradcam))
    return heatmap


def is_int(v):
    v = str(v).strip()
    return v == '0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()


def _exclude_layer(layer):
    if isinstance(layer, nn.Sequential):
        return True
    if not 'torch.nn' in str(layer.__class__):
        return True

    return False


def choose_tlayer(model):
    name_to_num = {}
    num_to_layer = {}
    for idx, data in enumerate(model.named_modules()):
        name, layer = data
        if _exclude_layer(layer):
            continue

        name_to_num[name] = idx
        num_to_layer[idx] = layer
        print(f'[ Number: {idx},  Name: {name} ] -> Layer: {layer}\n')

    print('\n<<-------------------------------------------------------------------->>')
    print('\n<<      You sholud not select [classifier module], [fc layer] !!      >>')
    print('\n<<-------------------------------------------------------------------->>\n')

    a = input(f'Choose "Number" or "Name" of a target layer: ')

    if a.isnumeric() == False:
        a = name_to_num[a]
    else:
        a = int(a)
    try:
        t_layer = num_to_layer[a]
        return t_layer
    except IndexError:
        raise NoIndexError('Selected index (number) is not allowed.')
    except KeyError:
        raise NoSuchNameError('Selected name is not allowed.')