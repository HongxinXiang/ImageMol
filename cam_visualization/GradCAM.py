import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from utils.gradcam_utils import cuda_available
from utils.gradcam_utils import save, choose_tlayer


class GradCAM():
    def __init__(self, img, model, gradcam_path, thresh=0.0, select_t_layer=False, class_index=None):
        assert img[1].shape[0] == 1
        self.img_show = img[0]
        self.img = img[1]
        self.model = model
        self.class_index = class_index
        self.select_t_layer = select_t_layer
        self.gradcam_path = gradcam_path
        self.thresh = thresh

        # Save outputs of forward and backward hooking
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        # find finalconv layer name
        if self.select_t_layer == False:
            finalconv_after = ['classifier', 'avgpool', 'fc']

            for idx, m in enumerate(self.model._modules.items()):
                if any(x in m for x in finalconv_after):
                    break
                else:
                    self.finalconv_module = m[1]

            # get a last layer of the module
            self.t_layer = self.finalconv_module[-1]
        else:
            # get a target layer from user's input
            self.t_layer = choose_tlayer(self.model)

        self.t_layer.register_forward_hook(forward_hook)
        self.t_layer.register_backward_hook(backward_hook)

    def __call__(self):
        # numpy to tensor and normalize
        self.input = self.img.reshape(-1, 3, 224, 224).cuda()

        output = self.model(self.input)

        if self.class_index == None:
            # get class index of highest prob among result probabilities
            self.class_index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][self.class_index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)

        if cuda_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = self.gradients['value']
        activations = self.activations['value']

        # reshaping
        weights = torch.mean(torch.mean(gradients, dim=2), dim=2)
        weights = weights.reshape(weights.shape[1], 1, 1)
        activationMap = torch.squeeze(activations[0])

        # Get gradcam
        gradcam = F.relu((weights * activationMap).sum(0))
        gradcam = cv2.resize(gradcam.data.cpu().numpy(), (224, 224))
        heatmap = save(gradcam, self.img_show, self.gradcam_path, thresh=self.thresh)

        return heatmap

