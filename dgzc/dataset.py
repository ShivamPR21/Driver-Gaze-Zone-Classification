import os
from typing import Any, Dict, Tuple

import face_recognition
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DGZCDataset(Dataset):

    def __init__(self, root:str,
                 size:Tuple[int, int] = (200, 200),
                 mode:str = 'train/auto_enc') -> None:
        super().__init__()

        assert(mode in ['train/auto_enc', 'train/classifier', 'inference'])
        self.root = root
        self.mode = mode
        self.train_dir = os.path.join(self.root, 'train')
        self.test_dir = os.path.join(self.root, 'test')
        self.size = size
        self.class_map:Dict[str, int] = {'Centerstack': 0,
                                         'Forward': 1,
                                         'Left_wing_mirror': 2,
                                         'Rearview_mirror': 3,
                                         'Right_wing_mirror': 4,
                                         'other': -1}

        self.images_path = []
        self.labels = []
        self.init_data()

    def init_data(self):

        if self.mode in ['train/auto_enc', 'train/classifier']:
            # Images in train directory
            for dir in os.listdir(self.train_dir):
                dir_path = os.path.join(self.train_dir, dir)

                images_path = [os.path.join(dir_path, img) for img in os.listdir(dir_path)]
                self.labels += [self.class_map[dir]]*len(images_path)
                self.images_path += images_path

        if self.mode in ['train/auto_enc', 'inference']:
            # Images in test directory
            images_path = [os.path.join(self.test_dir, img) for img in os.listdir(self.test_dir)]
            self.labels += [self.class_map['other']]*len(images_path)
            self.images_path += images_path


    def __getitem__(self, index: Any) -> Any:
        img_path = self.images_path[index] # Get image path
        image = face_recognition.load_image_file(img_path) # load image
        face_locs = face_recognition.face_locations(image)
        face_image = image
        if len(face_locs) > 0:
            top, right, bottom, left = face_locs[0] # get face location
            face_image = image[top:bottom, left:right] # get face centric image crop

        face_image = Image.fromarray(face_image)
        face_image = face_image.resize(self.size)
        face_image = np.asarray(face_image, dtype=np.float32)/255.
        face_image : torch.Tensor = torch.as_tensor(face_image, dtype=torch.float32).permute(2, 0, 1)

        target = torch.tensor(self.labels[index], dtype=torch.long)
        return face_image, target

    def __len__(self):
        return len(self.images_path)

class DGZCAutoEncoderDataset(DGZCDataset):

    def __init__(self, root: str, size: Tuple[int, int] = (200, 200), mode: str = 'train/auto_enc') -> None:
        super().__init__(root, size, mode)

class DGZCClassifierDataset(DGZCDataset):

    def __init__(self, root: str, size: Tuple[int, int] = (200, 200), mode: str = 'train/classifier') -> None:
        super().__init__(root, size, mode)

class DGZCInferenceDataset(DGZCDataset):

    def __init__(self, root: str, size: Tuple[int, int] = (200, 200), mode: str = 'inference') -> None:
        super().__init__(root, size, mode)
