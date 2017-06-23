'''
 usage:

'''
import sys
import os
import dlib
import glob
import cv2
import numpy as np
import getopt
import tensorflow as tf
from datetime import datetime
import time
sys.path.append("..")
sys.path.append("../..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# configurations
MAX_NUM_OBJECTS = 5

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

def download_tf_data():
  if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

def init_tf():
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph

def adaptive_resize(frame, size):
  # resize into 'size'
  t_w, t_h = size
  t_ratio = float(t_w)/t_h
  h, w, _ = frame.shape
  ratio = float(w)/h
  if ratio > t_ratio:
    # based on width
    new_w = t_w
    new_h = t_w/ratio
  else:
    # based on height
    new_h = t_h
    new_w = t_h*ratio

  return cv2.resize(frame, (int(new_w), int(new_h)))

def extract_object_region(frame, boxes, scores, thres=.5):
  cropped = np.zeros_like(frame)

  h, w, _ = frame.shape

  selected_rois = []
  for i in range(MAX_NUM_OBJECTS):
    if scores[i] < thres:
      break
    ymin, xmin, ymax, xmax = boxes[i]
    ymin, xmin, ymax, xmax = map(int, (ymin*h, xmin*w, ymax*h, xmax*w))

    selected_rois.append(frame[ymin:ymax, xmin:xmax, :])

  return selected_rois


def main():
  print(__doc__)

  default_width  = 1280
  default_height = 720
  default_width  = 640
  default_height = 480
  args, source = getopt.getopt(sys.argv[1:], 'o:w:h:v:', ['out=', 'width=', 'height=', 'video='])
  args = dict(args)
  print(args)

  # set parameters
  width = args.get('--width', default_width)
  height = args.get('--width', default_height)
  out_filepath = args.get('--out', './result.avi')
  try: cam_dev_id = source[0]
  except: cam_dev_id = 0
  video_filepath = args.get('--video', None)

  try: cam_dev_id = source[0]
  except: cam_dev_id = 0


#  cv2.namedWindow('edge')
#  cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
#  cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

#  cap.set(cv2.CAP_PROP_POS_FRAMES, start)

  # setup video
  if video_filepath:
    print('video_mode')
    cap = cv2.VideoCapture(video_filepath)
  else:
    print('camera_mode')
    cap = cv2.VideoCapture(int(cam_dev_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # check camera status
  fps = cap.get(cv2.CAP_PROP_FPS)
  print('fps:{}'.format(fps))

  fourcc = cv2.VideoWriter_fourcc(*'X264')
  out = cv2.VideoWriter(out_filepath, fourcc, 25.0, (1280,720))

  # init tf object detection api
  download_tf_data()
  detection_graph = init_tf()
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      # loop
      interval_track_list = []
      while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
          # end of file
          break
        else:
          frame = adaptive_resize(frame, (640, 360))
          frame = cv2.flip(frame, 1)
          #cv2.imshow('display', frame)

          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = frame.copy()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          begin = time.time()
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          end = time.time()
          interval_track_list.append(end - begin)
          if len(interval_track_list) > 20:
            interval_track_list.pop(0)
          print("{} secs, frame rate: {}".format((end - begin), 1.0/np.mean(interval_track_list)))
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

          cropped_list = extract_object_region(frame, boxes[0], scores[0])
          vis = np.hstack([frame, image_np])

          cv2.imshow('out', vis)
          for i, cropped in enumerate(cropped_list):
            cv2.imshow('cropped%02d'%i, cropped)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key & 0xFF == 27:
          break

  out.release()
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
