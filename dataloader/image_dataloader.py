import os
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, index=None, img_transformer=None, normalize=None, ret_index=False, args=None):
        '''
        :param names: image path, e.g. ["./data/1.png", "./data/2.png", ..., "./data/n.png"]
        :param labels: labels, e.g. single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer:
        :param normalize:
        :param args:
        '''
        self.args = args
        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ret_index = ret_index
        if index is not None:
            self.index = index
        else:
            self.index = []
            for filename in filenames:
                self.index.append(os.path.splitext(os.path.split(filename)[1])[0])

    def get_image(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        data = self.get_image(index)
        if self.normalize is not None:
            data = self.normalize(data)
        if self.ret_index:
            return data, self.labels[index], self.index[index]
        else:
            return data, self.labels[index]

    def __len__(self):
        return self.total


def load_filenames_and_labels_multitask(image_folder, txt_file, task_type="classification"):
    assert task_type in ["classification", "regression"]
    df = pd.read_csv(txt_file)
    index = df["index"].values.astype(int)
    labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    names = [os.path.join(image_folder, str(item) + ".png") for item in index]
    assert len(index) == labels.shape[0] == len(names)
    return names, labels


def get_datasets(dataset, dataroot, data_type="raw"):
    assert data_type in ["raw", "processed"]

    image_folder = os.path.join(dataroot, "{}/{}/224/".format(dataset, data_type))
    txt_file = os.path.join(dataroot, "{}/{}/{}_processed_ac.csv".format(dataset, data_type, dataset))

    assert os.path.isdir(image_folder), "{} is not a directory.".format(image_folder)
    assert os.path.isfile(txt_file), "{} is not a file.".format(txt_file)

    return image_folder, txt_file


def Smiles2Img(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    try:
        mol = Chem.MolFromSmiles(smis)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
        if savePath is not None:
            img.save(savePath)
        return img
    except:
        return None

