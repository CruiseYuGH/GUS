import torch.utils.data as data
from torchvision import transforms
import cv2
import image_utils
import random
from collections import Counter


class ABAWDataSet(data.Dataset):
    def __init__(self, data_path_list, label_path_list, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path_list[0]
        self.label_path = label_path_list[0]
        
        self.file_paths = []
        self.imges=[]
        self.label = []
        with open(self.label_path, 'r') as fin:
            tmp = fin.readlines()
            for demo in tmp:
                tmpp = demo.split(' ')
                if phase == 'train':
                    if tmpp[0] == 'train':
                        self.file_paths.append(self.data_path + tmpp[1])
                        self.imges.append(cv2.imread(self.data_path + tmpp[1])[:, :, ::-1])
                        self.label.append(int(tmpp[2]))
                else:
                    if tmpp[0] == phase:
                        self.file_paths.append(self.data_path + tmpp[1])
                        self.imges.append(cv2.imread(self.data_path + tmpp[1])[:, :, ::-1])
                        self.label.append(int(tmpp[2]))

        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise,image_utils.data_augment]
        if phase == 'train':
            self.transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                    transforms.RandomErasing(scale=(0.02,0.25))])
        else:
            self.transform =  transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])]) 
        c = Counter(self.label)
        print(dict(c))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = self.imges[idx]
        label = self.label[idx]
        if self.phase == 'train':
                if random.uniform(0, 1) > 0.5:
                    image = self.aug_func[0](image)
                if random.uniform(0, 1) > 0.5:
                    image = self.aug_func[1](image)
                if label in ["1","2"] and random.uniform(0, 1) > 0.5:
                    image = self.aug_func[2](image,0.5)

        if self.transform is not None:
            image = self.transform(image)
        return image, label, path, idx
    
    def get_labels(self):
        return self.label

