import os
import shutil
import argparse

total_event = 0
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1


def create_dir(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except:
            print("%s not existed and can not be created." % path)
            exit(1)


def begin(output_path):
    create_dir(output_path)
    create_dir(output_path + "/train")
    create_dir(output_path + "/train/signal")
    create_dir(output_path + "/train/background")
    create_dir(output_path + "/validation")
    create_dir(output_path + "/validation/signal")
    create_dir(output_path + "/validation/background")
    create_dir(output_path + "/check")
    create_dir(output_path + "/check/signal")
    create_dir(output_path + "/check/background")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input picture dir.", required=True)
    parser.add_argument("-n", "--num", type=int, help="choose the picture number.")
    parser.add_argument("-o", "--output_path", help="output picture path.")
    parser.add_argument("-t", "--type", help="Event type.")
    args = parser.parse_args()
    fileList = []
    for i in os.walk(args.input):
        fileList = i[2]
        rootdir = i[0]
    total_event = len(fileList)
    if args.output_path is None:
        output = args.input
    else:
        output = args.output_path
    begin(output)

    if args.type == 'b':
        event_type = 'background'
    else:
        event_type = 'signal'

    for Eventid in range(0, total_event):

        if Eventid < train_ratio * total_event:
            folder = 'train'
        elif (total_event * train_ratio) <= Eventid < (total_event * (train_ratio + valid_ratio)):
            folder = 'validation'
        else:
            folder = 'check'

        srcdir = os.path.join(rootdir, fileList[Eventid])
        desdir = os.path.join(output, folder, event_type)
        shutil.copy(srcdir, desdir)
