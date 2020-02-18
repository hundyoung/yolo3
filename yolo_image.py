import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body,box_iou
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import cv2 as cv
import math


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        # Initialize the parameters
        self.confThreshold = 0.5  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold
        self.inpWidth = 416  # Width of network's input image
        self.inpHeight = 416  # Height of network's input image

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # out_boxes, out_scores, out_classes = self.sess.run(
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [image.size[1], image.size[0]],
        #         K.learning_phase(): 0
        #     })
        prediction = self.yolo_model.predict(image_data)
        # temp = np.ndarray(prediction)
        # print(out_boxes_1)
        boxes = self.yolo_head(prediction,self.anchors,len(self.class_names),(image.size[1],image.size[0]))
        # boxes = self.postprocess(prediction, image.size,self.anchors)
        print('Found {} boxes for {}'.format(len(boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for top,left,bottom,right,box_class_scores,box_classes in boxes:
            box_classes=int(box_classes)
            predicted_class = self.class_names[box_classes]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            label = '{} {:.2f}'.format(predicted_class, box_class_scores)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[box_classes])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[box_classes])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, prediction, new_image_size,anchors):
        # num_anchors = len(anchors)
        anchors= np.reshape(anchors,[3,3, 2])
        num_layers = len(prediction)
        anchors_index = 0
        for l in range(num_layers):
            anchors_index+=3
            layer = prediction[l]
            grids=layer[0]
            grid_num = len(grids)
            input_shape=np.shape(grids)[:2]
            grids = np.reshape(grids,[grid_num, grid_num, 3, len(self.class_names) + 5])
            grids = self.sigmoid(grids)
            # print(grids)
            grid_width = new_image_size[0] / grid_num
            grid_height = new_image_size[1] / grid_num
            boxes=[]
            for i in range(grid_num):
                grid_row = grids[i]
                grid_y = i
                for j in range(grid_num):
                    grid = grid_row[j]
                    for k in range(len(grid)):
                        box = grid[k]
                        box_confidence = box[4]
                        box_class_probs = box[5:]
                        box_scores = box_confidence * box_class_probs
                        box_classes = np.argmax(box_scores, axis=-1)
                        box_class_scores =  np.max(box_scores, axis=-1)

                        if box_class_scores>0.6:
                            grid_x = j
                            tx, ty = box[0], box[1]
                            bx, by = grid_x + tx, grid_y + ty
                            # bw, bh = math.exp(box[2])*(anchors[l][k][0]/(grid_num*32)),math.exp(box[2])*(anchors[l][k][1]/(grid_num*32))
                            bw, bh = math.exp(box[2])*(anchors[l][k][0]/(grid_num)),math.exp(box[3])*(anchors[l][k][1]/(grid_num))

                            bx,by = (new_image_size[0]/grid_num)*bx,(new_image_size[1]/grid_num)*by
                            bw,bh = (new_image_size[0]/grid_num)*bw,(new_image_size[1]/grid_num)*bh
                            # box_yx = np.array([by,bx])
                            # box_hw = np.array([bh,bw])
                            # new_shape = np.round(input_shape * min(input_shape[0] / new_image_size[0],input_shape[1] / new_image_size[1]))
                            # offset = (input_shape - new_shape) / 2. / input_shape
                            # scale = input_shape / new_shape
                            # box_yx = (box_yx - offset) * scale
                            # box_hw *= scale
                            #
                            # box_mins = box_yx - (box_hw / 2.)
                            # box_maxes = box_yx + (box_hw / 2.)
                            # box_mins *= new_image_size[::-1]
                            # box_maxes *= new_image_size[::-1]
                            # top = by
                            # left = bx
                            # bottom = by+bh
                            # right = bx+bw

                            # boxes.append((box_mins[1:2],box_mins[0:1],box_maxes[1:2],box_maxes[0:1],box_classes,box_class_scores))
                            boxes.append((by,bx,by+bh,bx+bw,box_classes,box_class_scores))

            return boxes

    def yolo_head(self,feats,anchors, num_classes, image_shape, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        # num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        result = []
        boxes_list = []
        box_scores_list = []
        for l in range(len(feats)):
            anchor = anchors[anchor_mask[l]]
            anchors_tensor = np.reshape(anchor, [1, 1, 1, len(anchor), 2])
            feat = feats[l]
            grid_shape = np.shape(feat)[1:3] # height, width
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                            [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                            [grid_shape[0], 1, 1, 1])
            grid = np.concatenate((grid_x, grid_y),axis=-1)
            # grid = np.concatenate((grid_x, grid_y))
            feat = np.reshape(
                feat, [-1, grid_shape[0], grid_shape[1], 3, 80 + 5])


            box_confidence = self.sigmoid(feat[..., 4:5])
            box_class_probs = self.sigmoid(feat[..., 5:])
            # box_scores = box_confidence * box_class_probs
            # box_classes = np.argmax(box_scores, axis=-1)
            # box_class_scores = np.max(box_scores, axis=-1)
            # feat = np.insert(feat[...,:4],-1,values=box_classes,axis=-1)
            # feat = np.insert(feat,-1,values=box_class_scores,axis=-1)
            # feat = feat[feat[...,5:]>self.confThreshold]

            # grid = np.cast(grid, np.dtype(feat))



            input_shape = np.array([416,416])
            # input_shape = np.multiply(grid_shape,32)
            # Adjust preditions to each spatial grid point and anchor size.
            box_xy = (self.sigmoid(feat[..., :2]) + grid) / grid_shape[::-1]
            box_wh = np.exp(feat[..., 2:4]) * anchors_tensor / input_shape[::-1]


            box_yx = box_xy[..., ::-1]
            box_hw = box_wh[..., ::-1]
            # input_shape = input_shape
            # image_shape = image_shape

            new_shape = np.round(np.multiply(image_shape,min(input_shape / image_shape)))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = np.multiply((box_yx - offset), scale)
            box_hw *= scale

            box_mins = box_yx - (box_hw / 2.)
            box_maxes = box_yx + (box_hw / 2.)
            boxes = np.concatenate((
                box_mins[..., 0:1],  # y_min
                box_mins[..., 1:2],  # x_min
                box_maxes[..., 0:1],  # y_max
                box_maxes[..., 1:2]  # x_max
            ),axis=-1)
            # Scale boxes back to original image shape.
            boxes = np.multiply(boxes, np.concatenate((image_shape, image_shape)))
            box_scores = np.multiply(box_confidence,box_class_probs)
            #NMS

            boxes = np.reshape(boxes, [-1, 4])
            box_scores = np.reshape(box_scores, [-1, num_classes])
            boxes_list.append(boxes)
            box_scores_list.append(box_scores)
        boxes_list = np.concatenate(boxes_list, axis=0)
        box_scores_list = np.concatenate(box_scores_list, axis=0)
        for c in range(num_classes):
            scores = box_scores_list[:,c]
            # scores = np.reshape(np.shape(boxes_list)[0],1)
            boxes_class = np.insert(boxes_list,4, c,1)
            boxes_class = np.insert(boxes_class,4, scores,1)
            mask = scores>self.confThreshold
            boxes_class = boxes_class[mask]
            if len(boxes_class)>0:
                boxes_class=sorted(boxes_class,key=lambda x:x[-1],reverse=True)
                iou_list = np.array(boxes_class)
                # candidate_box = boxes_class[0]
                # iou_list = []
                # result.append(candidate_box)
                while len(iou_list)>0:
                    candidate_box= iou_list[0]
                    result.append(candidate_box)
                    if len(iou_list)==1:
                        break
                    iou_list = iou_list[1:]
                    b1 = K.variable(value=iou_list[:,:4])
                    b2 = K.variable(value=candidate_box[:4])
                    iou = K.eval(box_iou(b1,b2))
                    iou_mask = iou<=self.iou
                    iou_mask =np.concatenate(iou_mask,axis=0)
                    iou_list = iou_list[iou_mask]
        # return result


            #Without nms
            # boxes_class = np.concatenate((boxes,box_scores),axis=-1)[0]
            # for i in range(len(boxes_class)):
            #     grid_y_list = boxes_class[i]
            #     for j in range(len(grid_y_list)):
            #         grid = grid_y_list[j]
            #         for k in range(len(grid)):
            #             box = grid[k]
            #             for m in range(4,len(box)):
            #                 probability = box[m]
            #                 if probability>self.confThreshold:
            #                     result.append((box[0],box[1],box[2],box[3],m-4,probability))
        return result

        # if calc_loss == True:
        #     return grid, feats, box_xy, box_wh
        # return box_xy, box_wh, box_confidence, box_class_probs


img = '.\TestData\\bird.jpg'
image = Image.open(img)
yolo = YOLO()
r_image = yolo.detect_image(image)
r_image.show()
