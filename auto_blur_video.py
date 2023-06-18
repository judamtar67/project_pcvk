#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import time


class Detector:
    def __init__(self, model_path, name=""):
        self.graph = tf.Graph()
        self.model_path = model_path
        self.model_name = name
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            self.graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                self.graph_def.ParseFromString(f.read())
                tf.import_graph_def(self.graph_def, name='')
        print(f"{self.model_name} model is created..")

    def detect_objects(self, img, threshold=0.3):

        print(
            "{} : Object detection has started..".format(self.model_name))

        start_time = time.time()
        objects = []

        with tf.compat.v1.Session(graph=self.graph) as sess:

            rows = img.shape[0]
            cols = img.shape[1]
            image_np_expanded = np.expand_dims(img, axis=0)

            (num, scores, boxes,
                classes) = self.sess.run(
                    [self.sess.graph.get_tensor_by_name('num_detections:0'),
                     self.sess.graph.get_tensor_by_name('detection_scores:0'),
                     self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                     self.sess.graph.get_tensor_by_name('detection_classes:0')],
                feed_dict={'image_tensor:0': image_np_expanded})

            for i in range(int(num)):
                score = float(scores[0, i])
                if score > threshold:
                    obj = {}
                    obj["id"] = int(classes[0, i])
                    obj["score"] = score
                    bbox = [float(v) for v in boxes[0, i]]
                    obj["x1"] = int(bbox[1] * cols)
                    obj["y1"] = int(bbox[0] * rows)
                    obj["x2"] = int(bbox[3] * cols)
                    obj["y2"] = int(bbox[2] * rows)
                    objects.append(obj)

            print(f"{self.model_name} : {len(objects)} objects have been found ")
        end_time = time.time()
        print("{} : Elapsed time: {}".format(
            self.model_name, str(end_time - start_time)))

        return objects


# In[4]:


import os
import argparse
import cv2


def blurBoxes(image, boxes):
    for box in boxes:
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        sub = image[y1:y2, x1:x2]

        blur = cv2.blur(sub, (25, 25))

        image[y1:y2, x1:x2] = blur

    return image


def main(args):
    model_path = args.model_path
    threshold = args.threshold

    detector = Detector(model_path=model_path, name="detection")

    capture = cv2.VideoCapture(args.input_video)

    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(args.output_video, fourcc,
                                 20.0, (int(capture.get(3)), int(capture.get(4))))

    frame_counter = 0
    while True:
        _, frame = capture.read()
        frame_counter += 1

        if frame is None:
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        faces = detector.detect_objects(frame, threshold=threshold)

        frame = blurBoxes(frame, faces)

        cv2.imshow("Output", frame)

        if args.output_video:
            output.write(frame)
            print('Blurred video has been saved successfully at', args.output_video, 'path')

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image blurring parameters')

    parser.add_argument('-i',
                        '--input_video',
                        help='Path to your video',
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to .pb model',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output_video',
                        help='Output file path',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help='Face detection confidence',
                        default=0.7,
                        type=float)
    args = parser.parse_args()

    assert os.path.isfile(args.input_video), 'Invalid input file'

    if args.output_video:
        assert os.path.isdir(os.path.dirname(
            args.output_video)), 'No such directory'

    main(args)


# In[ ]:




