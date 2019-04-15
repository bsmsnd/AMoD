# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:06:06 2019

@author: sunhu
"""

import os 
import pickle


filename = '2019_4_14_20_11_56'
filename_dic = filename + '_fleet_dic.pkl'
filename_data = filename + '.txt'
dir_dic = os.path.join('log', filename_dic)
dir_data = os.path.join('log', filename_data)
request_dic = None
data_p = None
with open(dir_dic, 'rb') as f:
    request_dic = pickle.load(f)
with open(dir_data, "r") as f:
    data_p = f.readline()

print('hh')