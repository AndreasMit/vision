#!/usr/bin/env python3 

import sys
import rospy
import cv2
import numpy as np
import tensorflow as tf
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keras_segmentation.predict import predict
from keras_segmentation.predict import model_from_checkpoint_path
from segment_detector.msg import PREDdata

red = (153, 0, 18)
dim=(720, 480) 

class box_detector:

  def __init__(self):
    self.image_pub = rospy.Publisher("/Detection",Image,queue_size=10)
    self.contour_pub = rospy.Publisher("CDetection", Image, queue_size=10)
    self.box_pub = rospy.Publisher("/box", PREDdata, queue_size=1000)
    self.image_sub = rospy.Subscriber("/iris_demo/ZED_stereocamera/camera/left/image_raw",Image,self.callback) 
    self.bridge = CvBridge()
    self.box = PREDdata()

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    with graph.as_default():
    	pr, seg_img = predict( model=mdl, inp=cv_image)
    segimg = seg_img.astype(np.uint8)
    # cv2.imshow('maks',segimg)
    # cv2.waitKey(30)

    # create combo image
    red_mask = np.zeros(cv_image.shape, np.uint8)
    color = tuple(reversed(red))
    red_mask[:] = color
    red_mask = cv2.bitwise_and(segimg, red_mask)
    combo_image=cv2.addWeighted(cv_image, 1, red_mask, 0.35 ,1)

    #create box
    mask = cv2.inRange(segimg, (130, 130, 130), (255, 255, 255))
    kernel = np.ones((6, 6), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # masked = cv2.bitwise_and(gray, mask)
    # masked = np.zeros(cv_image.shape, np.uint8)
    # masked[:,:,0] = cv2.bitwise_and(cv_image[:,:,0], mask)
    # masked[:,:,1] = cv2.bitwise_and(cv_image[:,:,1], mask)
    # masked[:,:,2] = cv2.bitwise_and(cv_image[:,:,2], mask)
    # cv2.imshow('maks',masked)
    # cv2.waitKey(30)

    contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours_blk]
    if len(contours_blk) > 0:
      index = np.argmax(areas)
    
    # for i, c in enumerate(contours_blk):
    #   cv2.drawContours(masked, contours_blk, i, (255*((i+1)%3==0), 255*((i%3)==0), 255*((i+2)%3==0) ), 5)
    #   cv2.imshow('mask',masked)
    #   cv2.waitKey(30)

    if len(contours_blk) > 0 and areas[index] > 300 and areas[index] < 10000:
        # Box creation for the detected coastline
        box = cv2.minAreaRect(contours_blk[index])
        box = cv2.boxPoints(box)
        box = np.int0(box)
        cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 1)

        self.box.box_1 = box[0][:]
        self.box.box_2 = box[1][:]
        self.box.box_3 = box[2][:]
        self.box.box_4 = box[3][:]
    else:
        # print('out of bounds')
        self.box.box_1 = [0,0]
        self.box.box_2 = [0,0]
        self.box.box_3 = [0,0]
        self.box.box_4 = [0,0]

    self.box_pub.publish(self.box)

    try:
      #image with contour
      combo_ros_image = self.bridge.cv2_to_imgmsg(combo_image, "bgr8")
      combo_ros_image.header.stamp = data.header.stamp
      self.contour_pub.publish(combo_ros_image)
      
      #image with box
      ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
      ros_image.header.stamp = data.header.stamp
      self.image_pub.publish(ros_image)

    except CvBridgeError as e:
      print(e)



def main(args):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config = config)
  gpu = tf.test.gpu_device_name()

  # Check available GPU devices.
  print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
  
  global mdl
  mdl = model_from_checkpoint_path("./src/segment_detector/mobilenet")
  global graph
  graph = tf.get_default_graph()

  ic = box_detector()
  rospy.init_node('box_detector', anonymous=True)
  print('starting node')
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  # cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
