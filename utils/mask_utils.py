import numpy as np


def create_random_mask(shape=(512, 224, 224), mask_ratio=0.1):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    :param mask_ratio: the ratio of masked area
    :return:
    '''
    bsz, h, w = shape[0], shape[1], shape[2]
    mask_format = np.zeros(h * w)
    mask_format[: int(h * w * mask_ratio)] = 1

    mask_matrix = []
    for _ in range(bsz):
        np.random.shuffle(mask_format)
        mask_matrix.append(mask_format.reshape(h, w))
    mask_matrix = np.array(mask_matrix, dtype=int)

    return mask_matrix


def create_rectangle_mask(shape=(512, 224, 224), mask_shape=(16, 16)):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    :param mask_shape: The shape size of the masked area
    :return:
    '''
    bsz, h, w = shape[0], shape[1], shape[2]
    assert h == w
    xs = np.random.randint(w, size=bsz)
    ys = np.random.randint(h, size=bsz)

    mask_matrix = []
    for i in range(bsz):
        x = xs[i]
        y = ys[i]
        mask_format = np.zeros((h, w))
        mask_format[x: x + mask_shape[0], y: y + mask_shape[1]] = 1
        mask_matrix.append(mask_format)

    mask_matrix = np.array(mask_matrix, dtype=int)

    return mask_matrix
