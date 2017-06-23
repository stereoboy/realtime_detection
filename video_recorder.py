import numpy as np
import cv2
import glob
import sys
import os
import getopt

def main():

  default_width = 1280
  default_height = 720
  args, source = getopt.getopt(sys.argv[1:], 'o:w:h:', ['out=', 'width=', 'height='])
  args = dict(args)

  # set parameters
  width = args.get('--width', default_width)
  height = args.get('--width', default_height)
  out_filepath = args.get('--out', './captured.avi')
  try: cam_dev_id = source[0]
  except: cam_dev_id = 0

  # initialize camera
  cap = cv2.VideoCapture(int(cam_dev_id))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # check camera status
  fps = cap.get(cv2.CAP_PROP_FPS)
  print('fps:{}'.format(fps))

  #
  # sudo apt-get install ffmpeg x264 libx264-dev
  #
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(out_filepath, fourcc, fps, (width, height))
  out.set(cv2.CAP_PROP_FPS, fps)

  count = 0
  while(cap.isOpened()):

    ret, frame = cap.read()

    if ret:
      count += 1
      #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = cv2.flip(frame, 1)
      cv2.imshow('display', frame)
      out.write(frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key & 0xFF == 27:
      break

  out.release()
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
