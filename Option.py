#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:07:27 2018

@author: jiangyunzhang
"""


    Tree = np.zeros((TotalTime,TotalTime))
    Tree[0][0] = Initial
    for i in range(0,TotalTime-1):
        for j in range(0,TotalTime-1):
            if Tree[i][j] != 0:
                Tree[i][j+1] = Tree[i][j] * u
                Tree[i+1][j+1] = Tree[i][j] * d
        
        time = 6
        option = np.zeros((time,time))
        for i in range(0,time):
            for j in range(0,time):
                if j == 0: 
                    option[i][time-j-1] = np.maximum( bond[i][time-j-1]-100 , 0 )
                else:
                    for l in range(0,time-j):
                            option[l][time-j-1] = (option[l][time-j] * q + option[l+1][time-j] * (1-q) )/ (1+Tree[l][time-j-1]/2)**(Deltat*2)

PriceOption = option[0][0]