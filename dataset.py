import torch.utils.data as data

from os import listdir
from os.path import join, splitext
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def is_not_mask(filename):
    return 'mask' not in filename

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x) and is_not_mask(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input, target1, target2 = self.load_files(index)

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target1 = self.target_transform(target1)
        if self.target_transform:
            target2 = self.target_transform(target2)

        return input, target1, target2

    def load_files(self, index):
        input_filename = self.image_filenames[index]
        basename, ext = splitext(input_filename)
        target1_name = f'{basename}-mask1{ext}'
        target2_name = f'{basename}-mask2{ext}'

        input = load_img(input_filename)
        target1 = load_img(target1_name)
        target2 = load_img(target2_name)
        return input, target1, target2

    def __len__(self):
        return len(self.image_filenames)
