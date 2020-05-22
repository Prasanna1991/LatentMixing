import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from PIL import Image

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def get_dataLoader_mix(transformSequence, trans_aug,labelled = 500, batch_size = 8,
                       txtFilePath='Labels/',
                       pathDirData =  '/home/pkg2182/Workplace/part_of_all'):

    pathFileTrain_L =  txtFilePath + '/train_' + str(labelled) + '.txt'
    pathFileTrain_U =  txtFilePath + '/train_500_unlab.txt'
    validation =  txtFilePath + '/train_500_val_5000.txt'
    test =  txtFilePath + '/train_500_test_10000.txt'


    datasetTrainLabelled = DatasetGenerator_Mix(path=pathDirData, textFile=[pathFileTrain_L],
                                                    transform=trans_aug)
    datasetTrainUnLabelled = DatasetGenerator_Mix(path=pathDirData, textFile=[pathFileTrain_U],
                                                    transform=TransformTwice(trans_aug))
    datasetVal = DatasetGenerator_Mix(path=pathDirData, textFile=[validation],
                                                    transform=transformSequence)
    datasetTest = DatasetGenerator_Mix(path=pathDirData, textFile=[test],
                                                    transform=transformSequence)


    dataLoaderTrainLabelled = DataLoader(dataset=datasetTrainLabelled, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataLoaderTrainUnLabelled = DataLoader(dataset=datasetTrainUnLabelled, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return dataLoaderTrainLabelled, dataLoaderTrainUnLabelled, dataLoaderVal, dataLoaderTest

class DatasetGenerator_Mix(Dataset):
    def __init__(self, path, textFile, transform):
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        pathDatasetFile = textFile[0]
        fileDescriptor = open(pathDatasetFile, "r")
        line = True

        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                imagePath = os.path.join(path, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(float(i)) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('L')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None: imageData = self.transform(imageData)
        return imageData, imageLabel

    def __len__(self):
        return len(self.listImagePaths)
