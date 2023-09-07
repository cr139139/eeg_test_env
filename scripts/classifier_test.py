#! /usr/bin/env python3
import numpy as np
import cv2
import rospy
from std_msgs.msg import Float64MultiArray
from recording_nodes import EEGSUB

from joblib import load

clf = load('tsc_model.joblib')

neurone = EEGSUB()

rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(160)

import collections

predictions = collections.deque([0], 10)

while not rospy.is_shutdown():
    Xi = np.array(neurone.data)
    Xi = Xi.reshape((Xi.shape[0] // 30, 30))
    cov = Xi.T @ Xi / (Xi.shape[0] - 1)
    prob = clf.predict_proba(cov[np.newaxis, :, :])[0]
    arg_max = np.argmax(prob)
    if prob[arg_max] > 0.75:
        predictions.append(arg_max)
    print(clf.classes_, max(set(predictions), key=predictions.count))
    rate.sleep()
