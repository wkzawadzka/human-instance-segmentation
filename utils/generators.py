import tensorflow as tf
import cv2
from sklearn.utils import shuffle
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    ''' Basic data generator
        - provided file_paths loads images ans preprocess them

        Args:
        file_paths (list): List of file paths for input images.
        width (int): Width of the desired input image to U-Net (default is 32*20).*
        height (int): Height of the desired input image to U-Net (default is 32*14).*
        batch_size (int): Number of samples per batch (default is 32).
        shuffle_data (bool): Whether to shuffle the data at the end of each epoch (default is True).

        * For technical reasons, need to be a multiple of 32 to work with U-Net.
    '''

    def __init__(self, file_paths, width=32*20, height=32*14, batch_size=32, shuffle_data=True, rescale=True):
        self.file_paths = self.sort_file_paths(file_paths)
        self.masks_paths = [file.replace('img', 'semantic')
                            for file in self.file_paths]
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.width = width
        self.height = height
        self.indexes = np.arange(len(self.file_paths))
        if shuffle_data:
            self.indexes = shuffle(self.indexes)
        self.rescale = rescale

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_files = self.file_paths[start_idx:end_idx]
        y_batch_files = self.masks_paths[start_idx:end_idx]

        x_batch = self.preprocess(
            self.read(batch_files, flag=cv2.IMREAD_COLOR))
        y_batch = self.preprocess(
            self.read(y_batch_files, flag=cv2.IMREAD_GRAYSCALE, expand=True))

        return (x_batch, y_batch)

    def on_epoch_end(self):
        if self.shuffle_data:
            self.indexes = shuffle(self.indexes)

    def read(self, file_paths, flag, expand=False):
        if self.rescale:
            if expand:
                return np.array([np.expand_dims(cv2.resize(cv2.imread(file, flag), (self.width, self.height), interpolation=cv2.INTER_NEAREST), axis=-1) for file in file_paths])
            return np.array([cv2.resize(cv2.imread(file, flag), (self.width, self.height), interpolation=cv2.INTER_NEAREST) for file in file_paths])
        else:
            if expand:
                return np.array([np.expand_dims(cv2.imread(file, flag), axis=-1) for file in file_paths])
            return np.array([cv2.imread(file, flag) for file in file_paths])

    def preprocess(self, images):
        images = [image.astype(np.float32) / 255.0 for image in images]
        return np.array(images)

    def sort_file_paths(self, file_paths):
        return sorted(file_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))


class DataPatchGenerator(tf.keras.utils.Sequence):
    '''
        Sliding window - patches data generator
        - provided file_paths loads images to desired resolution and preprocess them
        - additionally each image is decomposed into patches of desired shape
        - patches are obtained using sliding window technique with stride of half of window size

        Note: in the end this generator was not used as it produces very unbalanced dataset (most of
        the patches represented just the background not human class) and trainining with such technique 
        did not proved useful for our case.

    '''

    def __init__(self, file_paths, window_width, window_height, image_width, image_height, batch_size=32, shuffle_data=True):
        self.file_paths = self.sort_file_paths(file_paths)
        self.masks_paths = [file.replace('img', 'semantic')
                            for file in self.file_paths]
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.window_width = window_width
        self.window_height = window_height
        self.width = image_width
        self.height = image_height
        self.indexes = np.arange(len(self.file_paths))
        if shuffle_data:
            self.indexes = shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_files = self.file_paths[start_idx:end_idx]
        y_batch_files = self.masks_paths[start_idx:end_idx]

        x_batch = self.preprocess(
            self.read(batch_files, flag=cv2.IMREAD_COLOR))
        y_batch = self.preprocess(
            self.read(y_batch_files, flag=cv2.IMREAD_GRAYSCALE, expand=True))
        x_patches, y_patches = self.extract_patches(x_batch, y_batch)
        return (x_patches, y_patches)

    def on_epoch_end(self):
        if self.shuffle_data:
            self.indexes = shuffle(self.indexes)

    def read(self, file_paths, flag, expand=False):
        if expand:
            return np.array([np.expand_dims(cv2.resize(cv2.imread(file, flag), (self.width, self.height), interpolation=cv2.INTER_NEAREST), axis=-1) for file in file_paths])
        return np.array([cv2.resize(cv2.imread(file, flag), (self.width, self.height), interpolation=cv2.INTER_NEAREST) for file in file_paths])

    def preprocess(self, images):
        images = [image.astype(np.float32) / 255.0 for image in images]
        return np.array(images)

    def sort_file_paths(self, file_paths):
        return sorted(file_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def extract_patches(self, images, masks):
        stride = (self.window_height//2, self.window_width//2)
        patches_per_image = []
        for img, mask in zip(images, masks):
            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            img_tensor = img_tensor[tf.newaxis, ...]
            mask_tensor = mask_tensor[tf.newaxis, ...]

            img_patches = tf.image.extract_patches(images=img_tensor, sizes=[1, self.window_height, self.window_width, 1], strides=[
                                                   1, stride[0], stride[1], 1], rates=[1, 1, 1, 1], padding='VALID')
            mask_patches = tf.image.extract_patches(images=mask_tensor, sizes=[1, self.window_height, self.window_width, 1], strides=[
                                                    1, stride[0], stride[1], 1], rates=[1, 1, 1, 1], padding='VALID')

            img_patches = tf.reshape(
                img_patches, [-1, self.window_height, self.window_width, img.shape[2]])
            mask_patches = tf.reshape(
                mask_patches, [-1, self.window_height, self.window_width])
            # so the model doesn't predict 0 all the time.
            non_zero_sum_indices = [index for index, patch in enumerate(
                mask_patches) if np.sum(patch == 1) != 0]

            patches_per_image.append((img_patches.numpy(
            )[non_zero_sum_indices], mask_patches.numpy()[non_zero_sum_indices]))

        # concatenate patches from all images
        all_patches_img = np.concatenate(
            [patches_img for patches_img, _ in patches_per_image])
        all_patches_mask = np.expand_dims(np.concatenate(
            [patches_mask for _, patches_mask in patches_per_image]), axis=-1)

        return all_patches_img, all_patches_mask


class DataYoloPatchGenerator(tf.keras.utils.Sequence):
    '''
        Similar to DataPatchGenerator but instead of all possible patches of input image, take only
        places indicated by yolo model as interesting.

        Note: yolo model with lower than normally probability threshold can be desirable.
        Note 2: images that file_paths point to are expected to be window_width x window_height as
                the resizing here has been ommited for speed.

        Args:
        file_paths (list): List of file paths for input images (not masks!).
    '''

    def __init__(self, yolo_model, file_paths, window_width, window_height, image_width, image_height, batch_size=32, shuffle_data=True):
        self.file_paths = self.sort_file_paths(file_paths)
        self.model = yolo_model  # with function .predict(image)
        self.masks_paths = [file.replace('img', 'semantic')
                            for file in self.file_paths]
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.window_width = window_width
        self.window_height = window_height
        self.width = image_width
        self.height = image_height
        self.indexes = np.arange(len(self.file_paths))
        if shuffle_data:
            self.indexes = shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_files = self.file_paths[start_idx:end_idx]
        y_batch_files = self.masks_paths[start_idx:end_idx]

        # read original images
        x_batch = self.preprocess(
            self.read(batch_files, flag=cv2.IMREAD_COLOR))
        y_batch = self.preprocess(
            self.read(y_batch_files, flag=cv2.IMREAD_GRAYSCALE))

        # get patches - region of intrest from yolo
        x_patches, y_patches = self.extract_yolo_patches(x_batch, y_batch)

        # if none
        if len(x_patches) == 0:
            return self.__getitem__(index + 1)
        return (x_patches, y_patches)

    def on_epoch_end(self):
        if self.shuffle_data:
            self.indexes = shuffle(self.indexes)

    def read(self, file_paths, flag):
        return np.array([cv2.imread(file, flag) for file in file_paths])

    def preprocess(self, images):
        images = [image.astype(np.float32) / 255.0 for image in images]
        return np.array(images)

    def sort_file_paths(self, file_paths):
        return sorted(file_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def extract_yolo_patches(self, images, masks):
        all_patches_img = []
        all_patches_mask = []
        for img, mask in zip(images, masks):
            boxes, _ = self.model.detect(img)
            for box in boxes:
                x, y, w, h = box
                if h < w or w == 0 or h == 0 or x < 0 or y < 0:
                    continue
                sub_img = cv2.resize(
                    img[y:y+h, x:x+w], (self.window_width, self.window_height), interpolation=cv2.INTER_NEAREST)
                sub_mask = np.expand_dims(cv2.resize(
                    mask[y:y+h, x:x+w], (self.window_width, self.window_height), interpolation=cv2.INTER_NEAREST), axis=-1)
                all_patches_img.append(sub_img)
                all_patches_mask.append(sub_mask)

        return np.array(all_patches_img), np.array(all_patches_mask)
