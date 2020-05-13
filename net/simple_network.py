import argparse
import json
import os
import time

import matplotlib.pyplot as plt
from keras.callbacks import *
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from my_picture_generator import MyImageDataGenerator


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


class Train(object):
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.check_generator = None
        self.train_epoch = 100
        self.exit_signal = False
        self.size = (128, 128)
        self.check_result = {}
        self.time = int(time.time())
        self.model_name = "SimpleNetwork"
        self.history = None
        self.batch_size = 128
        self.single_model = None
        self.multi_gpus = False

    def create_model(self, recovery_path=None):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(self.size[0], self.size[0], 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        if recovery_path:
            model.load_weights(recovery_path)

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        self.model = model

    def prepare_train_data(self, input_path):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,vertical_flip=True)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data

        self.train_generator = train_datagen.flow_from_directory(
            '%s/train' % input_path,  # this is the target directory
            target_size=(self.size[0], self.size[1]),  # all images will be resized to 150x150
            batch_size=self.batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        self.validation_generator = test_datagen.flow_from_directory(
            '%s/validation' % input_path,
            target_size=(self.size[0], self.size[1]),
            batch_size=self.batch_size,
            class_mode='binary')
        self.prepare_check_data(input_path)

    def prepare_check_data(self, input_path):
        check_datagen = MyImageDataGenerator(rescale=1. / 255)
        self.check_generator = check_datagen.flow_from_directory(
            '%s/check' % input_path,
            target_size=(self.size[0], self.size[1]),
            batch_size=self.batch_size,
            class_mode='binary')

    def signal_handler(self, signum, frame):
        print("Try to save train data. It may take a long time")
        self.exit_signal = True

    def check(self, output_path):
        count = 0
        while count < self.check_generator.samples:
            if count % 100 == 0:
                print("%s/%s" % (count, self.check_generator.samples))
            batch_x, batch_y, filename = self.check_generator.next()
            result = self.model.predict(batch_x)
            for i in range(0, len(result)):
                self.check_result[filename[i]] = {
                    "true_type": int(batch_y[i]),
                    "predict_type": float(result[i][0])
                }
            count += len(batch_x)
        open(output_path, "w").write(json.dumps(self.check_result))

    def train_epoch(self, r):
        self.train_epoch = r

    def train(self, output_path="."):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        path = "%s/%s-%s-{epoch:02d}-{val_acc:.4f}.h5" % (output_path, self.model_name, self.time)
        if self.multi_gpus:
            model_check_point = ParallelModelCheckpoint(self.single_model, path, monitor='val_acc', verbose=0,
                                                        save_best_only=False,
                                                        save_weights_only=True, mode='auto', period=1)
        else:
            model_check_point = ModelCheckpoint(path, monitor='val_acc', verbose=0, save_best_only=False,
                                                save_weights_only=True, mode='auto', period=1)
        tb = TensorBoard(log_dir='./logs/'+self.model_name+'log')
        #tb.validation_data, _, _ = self.check_generator.next()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, min_delta=1e-4,cooldown=5)
        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.train_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.batch_size,
            callbacks=[model_check_point,tb,reduce_lr])

    def save_acc_loss(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.title('Train and validation accurary')
        plt.legend()
        plt.savefig(self.model_name + 'acc.png')
        plt.figure()
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Train and validation loss')
        plt.legend()
        plt.savefig(self.model_name + 'loss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Picture data path.")

    sub_parser = parser.add_subparsers(dest='operate')

    train_parser = sub_parser.add_parser("train", help="Train the network.")
    train_parser.add_argument("-l", "--load", help="Load the network weights.")
    train_parser.add_argument("-o", "--output", help="Path to save network weights.")
    train_parser.add_argument("-n", "--epoch", default=50, type=int, help="Train epoch number.")

    valid_parser = sub_parser.add_parser("check", help="Valid the network.")
    valid_parser.add_argument("-l", "--load", required=True, help="Load the network weights.")
    valid_parser.add_argument("-o", "--output", help="Path to save check results.")

    args = parser.parse_args()

    t = Train()
    if args.operate == "train":
        t.train_epoch = args.epoch
        t.prepare_train_data(args.input)
        if args.load:
            t.create_model(args.load)
        else:
            t.create_model()
        if args.output:
            t.train(args.output)
        else:
            t.train()
            t.save_acc_loss()
    if args.operate == "check":
        t.prepare_check_data(args.input)
        t.create_model(args.load)
        t.check(args.output)
