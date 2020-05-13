import argparse

from keras import optimizers
from keras.applications import *
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.utils import multi_gpu_model
import simple_network
from focal_loss import *

class Nasnet(simple_network.Train):
    def __init__(self):
        super(Densenet169, self).__init__()
        self.model_name = "Nasnet"
        self.size = (128, 128)
        self.batch_size=64

    def create_model(self, total_model_weight_path=None, base_model_weight_path=None, bottleneck_weight_path=None,
                     input_layer=None):
        model = NASNetMobile(include_top=False, weights=None, input_tensor=None,
                            input_shape=(self.size[0], self.size[1], 3), pooling='max')
        x = model.output
        #x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(model.input, predictions)
        if total_model_weight_path:
            print("Load total weight from ", total_model_weight_path)
            model.load_weights(total_model_weight_path)
        model.summary()
        self.multi_gpus = True
        self.single_model = model
        #self.model = multi_gpu_model(model,gpus=2)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.Adam(lr=1e-3),
                           metrics=['accuracy'])

        # def freeze_layer(self, input_layer):
        #     if not input_layer:
        #         return
        #     input_layer = input_layer.split(":")
        #     if len(input_layer) != 2:
        #         raise RuntimeError("Wrong input trainable-layer.")
        #     start, end = tuple(input_layer)
        #         #     for layer in self.model.layers:
        #         #         layer.trainable = False
        #         #     if end and not start:
        #         #         for layer in self.model.layers[:int(end)]:
        #         #             print("Trainable: %s" % layer.name)
        #         #             layer.trainable = True
        #         #     if start and not end:
        #         #         for layer in self.model.layers[int(start):]:
        #         #             print("Trainable: %s" % layer.name)
        #         #             layer.trainable = True
        #         #     if start and end:
        #         #         for layer in self.model.layers[int(start):int(end)]:
        #         #             print("Trainable: %s" % layer.name)
        #         #             layer.trainable = True
        #         #     self.model.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Picture data path.")
    sub_parser = parser.add_subparsers(dest='operate')
    train_parser = sub_parser.add_parser("train", help="Train the network.")
    train_parser.add_argument("-l", "--load", default=None, help="Load the network weights.")
    train_parser.add_argument("--load-resnet", default=None, help="Load the network weights.")
    train_parser.add_argument("--load-top", default=None, help="Load the network weights.")
    train_parser.add_argument("--trainable-layer", default="0:100", help="Set the trainable layer number.")
    train_parser.add_argument("-o", "--output", default=".", help="Path to save network weights.")
    train_parser.add_argument("-n", "--epoch", default=50, type=int, help="Train epoch number.")
    valid_parser = sub_parser.add_parser("check", help="Valid the network.")
    valid_parser.add_argument("-l", "--load", required=True, help="Load the network weights.")
    valid_parser.add_argument("-o", "--output", default="result.json", help="Path to save check result.")
    args = parser.parse_args()
    t = Nasnet()
    if args.operate == "train":
        t.prepare_train_data(args.input)
        t.create_model(args.load, args.load_resnet, args.load_top, args.trainable_layer)
        t.train_epoch = args.epoch
        t.train(args.output)
        t.save_acc_loss()
    if args.operate == "check":
        t.prepare_check_data(args.input)
        t.create_model(args.load)
        t.check(args.output)
