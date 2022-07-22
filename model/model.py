import math
import torch.nn as nn
from model.cnn_model_utils import load_model, get_support_model_names


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ImageMol(nn.Module):
    def __init__(self, baseModel, jigsaw_classes, label1_classes, label2_classes, label3_classes):
        super(ImageMol, self).__init__()

        assert baseModel in get_support_model_names()

        self.baseModel = baseModel

        self.embedding_layer = nn.Sequential(*list(load_model(baseModel).children())[:-1])

        self.bn = nn.BatchNorm1d(512)

        self.jigsaw_classifier = nn.Linear(512, jigsaw_classes)
        self.class_classifier1 = nn.Linear(512, label1_classes)
        self.class_classifier2 = nn.Linear(512, label2_classes)
        self.class_classifier3 = nn.Linear(512, label3_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(x.size(0), -1)

        x1 = self.jigsaw_classifier(x)
        x2 = self.class_classifier1(x)
        x3 = self.class_classifier2(x)
        x4 = self.class_classifier3(x)

        return x, x1, x2, x3, x4


# to discriminate rationality
class Matcher(nn.Module):
    def __init__(self):
        super(Matcher, self).__init__()
        self.fc = nn.Linear(512, 2)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


# initializing weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
