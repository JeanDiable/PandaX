from keras import optimizers
import simple_network
import argparse
from keras.models import Sequential, Model
from keras.applications import resnet50
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout


class ResNet50Train(simple_network.Train):
    def __init__(self):
        super(ResNet50Train, self).__init__()
        self.train_data_dir = None
        self.model_name = "Resnet50"
        self.size = (128, 128)

    def create_model(self, total_model_weight_path=None, base_model_weight_path=None, bottleneck_weight_path=None,
                     input_layer=None):
        model = resnet50.ResNet50(include_top=False, weights=None, input_tensor=None,
                                  input_shape=(self.size[0], self.size[1], 3))
        if total_model_weight_path:
            x = model.output
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(model.input, predictions)
            print("Load total weight from ", total_model_weight_path)
            model.load_weights(total_model_weight_path)
        else:
            if base_model_weight_path:
                print("Load vgg weight from ", base_model_weight_path)
                model.load_weights(base_model_weight_path)
            else:
                pass
                # raise RuntimeError("VGG Weight File Missing!")
            x = model.output
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(model.input, predictions)
        self.model = model
        # self.freeze_layer(input_layer)
        self.model.summary()
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                           metrics=['accuracy'])

        # def freeze_layer(self, input_layer):
        #     if not input_layer:
        #         return
        #     input_layer = input_layer.split(":")
        #     if len(input_layer) != 2:
        #         raise RuntimeError("Wrong input trainable-layer.")
        #     start, end = tuple(input_layer)
        #     for layer in self.model.layers:
        #         layer.trainable = False
        #     if end and not start:
        #         for layer in self.model.layers[:int(end)]:
        #             print("Trainable: %s" % layer.name)
        #             layer.trainable = True
        #     if start and not end:
        #         for layer in self.model.layers[int(start):]:
        #             print("Trainable: %s" % layer.name)
        #             layer.trainable = True
        #     if start and end:
        #         for layer in self.model.layers[int(start):int(end)]:
        #             print("Trainable: %s" % layer.name)
        #             layer.trainable = True
        #     self.model.summary()


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
    t = ResNet50Train()
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
