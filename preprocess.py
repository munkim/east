import os

# Prepare dataset


def get_record(path_dataset):
    # Convert to record file by running mxnet/tools/im2rec.py
    if os.path.exists('.rec') is False:
        MXNET_HOME = '/Users/apple/mxnet'
        os.system("python %s/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 --train-ratio=0.8 --test-ratio=0.2 "
                  "./data/icdar data/%s" % (MXNET_HOME, path_dataset))  # Outputs icdar_train.idx & icdar_test.idx
        os.system("python %s/tools/im2rec.py --num-thread=4 --pass-through=1 "
                  "./data/icdar data/%s" % (MXNET_HOME, path_dataset))  # Outputs icdar_train.rec & icdar_test.rec
