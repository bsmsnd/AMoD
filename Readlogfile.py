# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:06:06 2019

@author: sunhu
"""

import os 
import pickle


filename = '2019_4_14_17_8_1_fleet_dic.pkl'
dir_log = os.path.join('log', filename)
request_dic = None
with open(dir_log, 'rb') as f:
    request_dic = pickle.load(f)

