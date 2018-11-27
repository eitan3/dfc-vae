import numpy as np
import mxnet as mx
import random
from config_celeba_cnn import input_shape
import glob
import cv2
import os


def GetData(ctx, data_folder):
    subdirs = [os.path.join(data_folder, o) for o in os.listdir(data_folder)
               if os.path.isdir(os.path.join(data_folder,o))]

    train_split = subdirs[: int(len(subdirs) * 0.95)]
    test_split = subdirs[int(len(subdirs) * 0.95):]
    train_iter = CelebAIter(ctx, input_shape, train_split)
    test_iter = CelebAIter(ctx, input_shape, test_split)
    return train_iter, test_iter


class CelebAIter(mx.io.DataIter):
    def __init__(self, ctx, data_shapes, data_array):
        super(CelebAIter, self).__init__()
        self.ctx = ctx
        self.data_shapes = data_shapes
        self.batch_size = data_shapes[0]

        self.data_array = data_array
        self.num_examples = len(data_array)

        self.num_batches = self.num_examples / self.batch_size
        self.cur_batch = 0
        self.flip = False
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        random.shuffle(self.data_array)
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    def next(self):
        if self.cur_batch < self.num_batches:
            data = np.zeros(self.data_shapes, dtype=np.float32)

            for i in range(self.batch_size):
                index = self.cur_batch * self.batch_size + i
                indevidual_folder = self.data_array[index]
                indevidual_imgs = glob.glob(indevidual_folder + "/*.jpg")
                img_filename = random.choice(indevidual_imgs)
                img_filename = img_filename.replace('\\', '/')
                img_anno = img_filename[:-3]+"txt"

                with open(img_anno) as f:
                    anno_content = f.readlines()
                # you may also want to remove whitespace characters like `\n` at the end of each line
                anno_content = [x.strip() for x in anno_content]
                bbox = anno_content[0].split()[1:]
                bbox = [int(b) for b in bbox]

                img = cv2.imread(img_filename)
                img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
                img = cv2.resize(img, (input_shape[2], input_shape[2]), interpolation=cv2.INTER_LINEAR)
                img = img / 255.0 * 2.0 - 1.0
                img = np.swapaxes(img, 0, 2)
                img = np.swapaxes(img, 1, 2)
                img = img.astype('float32')

                data[i, :, :, :] = img[:]
                del img

            self.cur_batch += 1
            return mx.nd.array(data, ctx=self.ctx), None
        else:
            self.reset()
            raise StopIteration
