import time
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from utils.my_picture_generator import MyImageDataGenerator
import json
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

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
        self.batch_size = 64

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
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

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

    def prepare_check_data(self, input_path):
        check_datagen = MyImageDataGenerator(rescale=1. / 255)
        self.check_generator = check_datagen.flow_from_directory(
            '%s/check' % input_path,
            target_size=(self.size[0], self.size[1]),
            batch_size=self.batch_size,
            class_mode='binary')

    def convert_row(self):
        row = np.zeros((self.size[0], self.size[1], 3))
        for i in range(0, len(self.tree.cluster_xzx)):
            location_x = int(self.tree.cluster_xzx[i])
            location_z = int(self.tree.cluster_xzz[i])
            location_x += int(self.size[0] / 2)
            location_z += int(self.size[1] / 2)
            if not (0 <= location_x < self.size[0] and 0 <= location_z < self.size[1]):
                continue
            row[location_x, location_z, 0] = self.tree.cluster_xze[i]
        for i in range(0, len(self.tree.cluster_yzy)):
            location_y = int(self.tree.cluster_yzy[i])
            location_z = int(self.tree.cluster_yzz[i])
            location_y += int(self.size[0] / 2)
            location_z += int(self.size[1] / 2)
            if not (0 <= location_y < self.size[0] and 0 <= location_z < self.size[1]):
                continue
            row[location_y, location_z, 1] = self.tree.cluster_yze[i]
        return row

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
        model_check_point = ModelCheckpoint(path, monitor='val_acc', verbose=0, save_best_only=False,
                                            save_weights_only=True, mode='auto', period=1)
        """
        print("Training %s in %s." % (self.model_name, self.time))
        print "self.train_generator.samples is :", self.train_generator.samples
        samples_per_epoch=self.train_generator.samples//32
        print "samples_per_epoch is :",  samples_per_epoch
        """
        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples//self.batch_size,
            epochs=self.train_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples//self.batch_size,
            callbacks=[model_check_point])
        
    def save_acc_loss(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range( len(acc) )
        plt.plot(epochs , acc, label = 'Training acc')
        plt.plot(epochs , val_acc, label = 'Validation acc')
        plt.title('Train and validation accurary')
        plt.legend()
        plt.savefig( self.model_name+'acc.png')
        plt.figure()
        plt.plot(epochs , loss, label = 'Training loss')
        plt.plot(epochs , val_loss, label = 'Validation loss')
        plt.title('Train and validation loss')
        plt.legend()
        plt.savefig( self.model_name+'loss.png' )

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
