#!/usr/bin/env python3

import os
from Configuration import Configer
import time
import datetime
import cv2
import numpy as np
import uuid
import json

import functools
import logging
import collections

logger = logging.getLogger(Configer.LOGGING_FILE_PATH)
logger.setLevel(logging.INFO)


def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow as tf
    import model
    from icdar import restore_rectangle
    import lanms
    from eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        ret.update(get_host_info())
        return ret


    return predictor


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu



def detect_text_regions_from_image (image_path):
    if not os.path.exists(Configer.EAST_MODEL_CHECKPOINT_FOLD):
        raise RuntimeError(
            'Checkpoint `{}` not found, please '.format(
                Configer.EAST_MODEL_CHECKPOINT_FOLD))
    #load test images
    marked_result_path, result_json_path = generate_result_file_path(
            image_path)
    img = cv2.imread(image_path, 1)
    rst = get_predictor(Configer.EAST_MODEL_CHECKPOINT_FOLD)(img)

    # save illustration
    cv2.imwrite(marked_result_path, draw_illu(img.copy(), rst))

    # save json data
    with open(result_json_path, 'w') as f:
      json.dump(rst, f)

    del img


def generate_result_file_path(image_path):
    image_fold, image_file_name = os.path.split(image_path)
    image_file_name, image_file_ext = os.path.splitext(image_file_name)
    marked_result_path = os.path.join(image_fold,'marked_'+image_file_name,
                             image_file_ext)
    result_json_path = os.path.join(image_fold, 'result.json')
    return marked_result_path,result_json_path

if __name__ == '__main__':
  # global checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_path', description = 'set path '
                                                            'for image to be '
                                                            'processed.')
    args = parser.parse_args()
    detect_text_regions_from_image(args[0])

    # parser.add_argument('--checkpoint-path', default=checkpoint_path)
    # parser.add_argument('--debug', action='store_true')
    # args = parser.parse_args()
    # checkpoint_path = args.checkpoint_path
    pass
    #main()

