import sys
sys.path.append("../")
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from cam_visualization.GradCAM import GradCAM
from model.cnn_model_utils import load_model


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_image(img_path, normalize, img_transformer):
    img = Image.open(img_path).convert('RGB')
    img_trans = img_transformer(img)
    img_show = img_trans.numpy().transpose(1, 2, 0)
    return img_show, normalize(img_trans)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch GradCAM')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--resume', required=True, type=str, metavar='PATH')
    parser.add_argument('--img_path', type=str, required=True, help='path to image file')
    parser.add_argument('--gradcam_save_path', type=str, required=True, help='path to saved gradcam')
    parser.add_argument('--thresh', type=float, default=0.0, help='thresh of gradcam')
    args = parser.parse_args()

    # 1. initialize model
    model = load_model(args.image_model, imageSize=224, num_classes=2)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint['epoch']
            arch = checkpoint['arch']
            print("resume model info: epoch: {}; arch: {}".format(epoch, arch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model = model.cuda()
    model.eval()

    # 2. initialize data
    img_cuda = torch.FloatTensor(1, 3, 224, 224).cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transformer = [transforms.CenterCrop(224), transforms.ToTensor()]
    img_show, img = get_image(args.img_path, normalize, transforms.Compose(img_transformer))
    img_cuda.copy_(torch.unsqueeze(img, 0))

    # 3. run gradcam
    gradcam_obj = GradCAM(img=(img_show, img_cuda), model=model, gradcam_path=args.gradcam_save_path, thresh=args.thresh)
    heatmap = gradcam_obj()
    print("execute completed.")

