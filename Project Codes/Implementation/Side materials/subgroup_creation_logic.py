# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:55:19 2021

@author: hatta
"""

import pandas as pd
import numpy as np

sens_attr = ['age','gender']

#structure i need:
privileged_groups = [{'sex': 1, 'age': 1}]
unprivileged_groups = [{'sex': 0, 'age': 0}]

#find all the positive combinations for the sensitive subgroups
sens_groups = []
for i in range(0,2):
    for k in range(0,2):
        sens_g = [{sens_attr[0]:k, sens_attr[1]:i}]
        sens_groups.append(sens_g)
        print(sens_g)
print(sens_groups)


for i in range(0,3):
    for k in range(0,3):
        if (k+1+i)<=3:
            unprivileged_group = sens_groups[(k)]
            privileged_group = sens_groups[(k+1+i)]
            concatted = str(unprivileged_group)+str(privileged_group)
            print("result:", concatted)
        else:
            continue
