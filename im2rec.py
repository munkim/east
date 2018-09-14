import os
import sys
import argparse
import re
import random
import time
import cv2
import traceback
import mxnet as mx
import codecs

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


"""
    TO DO:
        1) Refactor to support text detection labeling.
            - Currently, the label of each text (not the label of bbox) is random.
            - To do so, first, create word2idx file.
            - Then, change the label by looking at the vocab word2idx.  
"""


def icdar2013_list_generator(root):
    """
    Get image/label file and create a list generator with 'yield' function.
    The root needs /images and /labels.
    The file naming under sub-dirs: ('img_1.jpg' in /images and 'gt_img_1.txt' in /labels)
    The format of the label file: (x1, y2, ..., 'transcription')
    """
    i = 0
    # The root to data should include two directories with /images and /labels
    assert 'images' or 'labels' not in os.listdir(root), "The /images or /labels directory does not exist."
    for image in os.listdir(os.path.join(root, 'images')):
        for label in os.listdir(os.path.join(root, 'labels')):
            if image.split('.')[0] in label:
                with codecs.open(os.path.join(root, 'labels', label), 'r', encoding="utf-8-sig") as f:
                    lines = f.readlines()
                    A = 2 # A = number of headers, B = width of each object label
                    B = 5  # For documentations: https://mxnet.io/versions/master/api/python/image/image.html
                    image_path = os.path.join(root, 'images', image)
                    img = cv2.imread(image_path, 0)
                    img_height, img_width = img.shape[:2]
                    list_line = '%d\t%d\t%d\t%d\t%d\t' % (i, A, B, img_width, img_height)
                    i += 1
                    j = 0
                    for line in lines:
                        line = line.split(', ')
                        label = re.findall(r',?([^,]+)(?:,|\r\n)', line[-1])
                        bbox = line[:-1]
                        for k, axis in enumerate(bbox):
                            # print ()
                            bbox[k] = float(axis)/img_width if k % 2 == 0 else float(axis)/img_height
                            bbox[k] = str(round(bbox[k], 3))
                        bbox = '\t'.join(bbox)
                        # label = re.findall('"([^"]*)"', line[-1])[0]
                        list_line += '%d\t%s\t' % (j, bbox)
                        j += 1

                list_line += '%s\n' % image_path
                yield list_line
                break
        if i == 0:  # If there is no
            raise ValueError("There is no file in /label that contains the name %s." % file)


def icdar2015_list_generator(root):
    """
    Get image/label file and create a list generator with 'yield' function.
    The root needs /images and /labels.
    The file naming under sub-dirs: ('img_1.jpg' in /images and 'gt_img_1.txt' in /labels)
    The format of the label file: (x1, y2, ..., 'transcription')
    """
    i = 0
    # The root to data should include two directories with /images and /labels
    assert 'images' or 'labels' not in os.listdir(root), "The /images or /labels directory does not exist."
    for image in os.listdir(os.path.join(root, 'images')):
        for label in os.listdir(os.path.join(root, 'labels')):
            if image.split('.')[0] in label:
                with codecs.open(os.path.join(root, 'labels', label), 'r', encoding="utf-8-sig") as f:
                    lines = f.readlines()
                    A = 4  # A = number of headers, B = width of each object label
                    B = 9  # For documentations: https://mxnet.io/versions/master/api/python/image/image.html
                    image_path = os.path.join(root, 'images', image)
                    img = cv2.imread(image_path, 0)
                    img_height, img_width = img.shape[:2]
                    list_line = '%d\t%d\t%d\t%d\t%d\t' % (i, A, B, img_width, img_height)
                    i += 1
                    j = 0
                    for line in lines:
                        line = re.findall(r',?([^,]+)(?:,|\r\n)', line)
                        bbox = line[:-1]
                        for k, axis in enumerate(bbox):
                            bbox[k] = float(axis)/img_width if k % 2 == 0 else float(axis)/img_height
                            bbox[k] = str(round(bbox[k], 3))
                        bbox = '\t'.join(bbox)
                        list_line += '%d\t%s\t' % (j, bbox)
                        j += 1

                list_line += '%s\n' % image_path
                yield list_line
                break
        if j == 0:  # If there is no
            raise ValueError("There is no file in /label that contains the name %s." % image)

def write_list(list_generator, path_out):
    with open(path_out, 'w') as f_out:
        for i, line in enumerate(list_generator):
            f_out.write(line)

def make_list_files(args):
    if args.dataset == 'ICDAR2013':
        list_generator = icdar2013_list_generator(args.root)
    elif args.dataset == 'ICDAR2015':
        list_generator = icdar2015_list_generator(args.root)
    else:
        raise RuntimeError("%s is incorrect not available. Has to be either ICDAR2013 or ICDAR2015." % args.dataset)
    list_generator = list(list_generator)
    N = len(list_generator)

    if args.shuffle is True:
        random.seed(100)
        random.shuffle(list_generator)
    if args.test_ratio:
        write_list(list_generator[:int(args.test_ratio * N)],
                   os.path.join(args.root, 'test.lst'))
        print ("Successfully created test.lst")
    if args.test_ratio + args.train_ratio < 1.0:
        write_list(list_generator[int((args.test_ratio + args.train_ratio) * N):],
                   os.path.join(args.root, 'valid.lst'))
        print ("Successfully created valid.lst")
    write_list(list_generator[int(args.test_ratio * N):int((args.test_ratio + args.train_ratio) * N)],
               os.path.join(args.root, 'train.lst'))
    print ("Successfully created train.lst")


def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 3:
                print('lst should at least has three parts, but only has %s parts for %s' %(line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [i for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' %(line, e))
                continue
            yield item


def image_encode(args, i, item, q_out):
    fullpath = item[1]
    if len(item) > 3 and args.pack_label:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)
    # print (header)
    # exit()

    if args.pass_through:
        try:
            with open(fullpath, 'rb') as fin:
                img = fin.read()
            s = mx.recordio.pack(header, img)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return

    try:
        img = cv2.imread(fullpath, args.color)
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % fullpath)
        q_out.put((i, None, item))
        return

    if args.center_crop:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) // 2
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) // 2
            img = img[:, margin:margin + img.shape[0]]
    if args.resize:
        if img.shape[0] > img.shape[1]:
            newsize = (args.resize, img.shape[0] * args.resize // img.shape[1])
        else:
            newsize = (img.shape[1] * args.resize // img.shape[0], args.resize)
        img = cv2.resize(img, newsize)

    try:
        # print (header)
        # exit()
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return

def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)

def write_worker(args, fname, q_out):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(args.root, fname_idx),
                                           os.path.join(args.root, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)
            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1

def make_rec_files(args, fname, list_generator):
    # -- write_record -- #
    if args.num_thread > 1 and multiprocessing is not None:
        print ("Number of Thread: {}".format(args.num_thread))
        q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
        q_out = multiprocessing.Queue(1024)
        read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                        for i in range(args.num_thread)]
        for p in read_process:
            p.start()
        write_process = multiprocessing.Process(target=write_worker, args=(args, fname, q_out))
        write_process.start()

        for i, item in enumerate(list_generator):
            q_in[i % len(q_in)].put((i, item))
        for q in q_in:
            q.put(None)
        for p in read_process:
            p.join()

        q_out.put(None)
        write_process.join()
        print ("Record files successfully created!")
    else:
        print('multiprocessing not available, fall back to single threaded encoding')
        try:
            import Queue as queue
        except ImportError:
            import queue
        q_out = queue.Queue()
        fname = os.path.basename(fname)
        fname_rec = os.path.splitext(fname)[0] + '.rec'
        fname_idx = os.path.splitext(fname)[0] + '.idx'
        record = mx.recordio.MXIndexedRecordIO(os.path.join(args.root, fname_idx),
                                               os.path.join(args.root, fname_rec), 'w')
        cnt = 0
        pre_time = time.time()
        for i, item in enumerate(list_generator):
            image_encode(args, i, item, q_out)
            if q_out.empty():
                continue
            _, s, _ = q_out.get()
            record.write_idx(item[0], s)
            if cnt % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', cnt)
                pre_time = cur_time
            cnt += 1
        print ("Record files successfully created!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', default='./data', help='Path to ICDAR 13/15 data directory with /images and /labels folders.')
    parser.add_argument('-dataset', default='ICDAR2015', help='ICDAR2013 or ICDAR2015')
    parser.add_argument('-train-ratio', type=float, default=0.9,  help='Ratio of images to use for training.')
    parser.add_argument('-test-ratio', type=float, default=0, help='Ratio of images to use for testing.')
    parser.add_argument('-shuffle', type=bool, default=True,
                        help='If this is set as True, im2rec will randomize the image order in <prefix>.lst')
    parser.add_argument('-num_thread', type=int, default=10, help='Number of Thread')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--pass-through', type=bool, default=False,
                        help='whether to skip transformation and save image as is')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, '
                             'original images will be packed by default.')
    rgroup.add_argument('--center-crop', type=bool, default=False,
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. '
                             'order of images will be different from the input list if >1. '
                             'the input list will be modified to match the resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image. '
                             '1: Loads a color image. Any transparency of image will be neglected. It is the default flag.'
                             '0: Loads image in grayscale mode. '
                             '-1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', type=bool, default=True,
                        help='Whether to also pack multi dimensional label in the record file')

    args = parser.parse_args()

    print ("\nMaking list files from %s..." % (args.root))
    make_list_files(args)  # Make .lst file
    for file in os.listdir(args.root):
        if '.lst' in file:
            print ("\nReading list from %s..." % file)
            list_generator = read_list(os.path.join(args.root, file))  # Read .lst file
            print ("\nMaking record files...")
            t = time.time()
            make_rec_files(args, os.path.join(args.root, file), list_generator)  # Make .rec file
            print ("TIME REC FILE {}".format(time.time() - t))

if __name__=="__main__":
    main()