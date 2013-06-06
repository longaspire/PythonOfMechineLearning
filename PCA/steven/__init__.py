# -*- coding:utf-8 -*-

# @author: StevenLee @contact: StuNo.11221078 Phone:13646823639

import os
from numpy import *
import numpy as np
from copy import deepcopy
from matplotlib.mlab import PCA
import Gnuplot, Gnuplot.funcutils

def writeCharList(charlist):
    fp = open ('optdigits-orig_3.tra', 'a')
    for item in charlist:
        fp.write(item)
    fp.write('\n')
    fp.close()
        
def getThreeSamples():
    row_count = 33
    k = 0
    charList = []
    
    fileHandle = open ('optdigits-orig.tra')  
    fileList = fileHandle.readlines()
    if os.path.exists('optdigits-orig_3.tra') is True:
        os.remove('optdigits-orig_3.tra')
        
    for fileLine in fileList:
        row_count -= 1
        if row_count != 0:
            charList.append(fileLine)  # as a buffer of the matrix
        else:
            row_count = 33
            if fileLine == ' 3\n':
                k += 1
                writeCharList(charList) # write the three sample to the file
            charList = []
    fileHandle.close()
    print 'have already got all the Threes from original files.'
    
def getMatrixs():
    feature, featureMatrix = [], []
    
    fileHandle = open ('optdigits-orig_3.tra')  
    fileList = fileHandle.readlines()
    for fileLine in fileList:
        line = fileLine.rstrip()
        #print line
        if line != "":
            feature += [int(x) for x in line]
        else:
            #print len(feature)
            featureMatrix.append(feature)
            feature = []
    
    #print len(featureMatrix)
    dataMatrix = np.array(featureMatrix)
    #print dot(dataMatrix,dataMatrix.T)  
    print 'have already transformed the samples into matrixs.'
    print shape(dataMatrix.T)
    #fp = open ('result', 'a')
    #fp.write(str(featureMatrix))
    #fp.close()
    return dataMatrix.T

def myPCA(data):
    
    n, m = data.shape
    if n < m:
        raise RuntimeError('we assume data in a is organized with num_rows>num_cols')

    dataT = data.T - data.mean(1)
    data = dataT.T

    U, s, Vh = np.linalg.svd(data, full_matrices=False)
    
    print U,s,Vh

    U_d2 = U[:,0:2].T
    
    Y = np.dot(U_d2,data).T
    
    return Y
    
    print 'have already finished the process of PCA.'
    
print 'PCA solving----------------------------'

print 'Get all Threes from original files:'
getThreeSamples()

print 'Transform the samples into matrixs:'
SVDMatrix = getMatrixs()
print shape(SVDMatrix)

print 'Sigular Value Decomposition:'
result = myPCA(SVDMatrix)

x = []
y = []
data = []

for item in result:
    x.append(item[0]) #add the column in result.Y
    y.append(item[1])
    data.append((item[0],item[1]))

gp = Gnuplot.Gnuplot() # define a GNUplot object
gp('set data style lines')
gp.title("Principal Component Analysis @author: StevenLee")
gp.xlabel('First Principal Component')
gp.ylabel('Second Principal Component')
 
realDataplot = Gnuplot.PlotItems.Data(data, with_="points pointtype 7 pointsize 0")
firstPline = Gnuplot.PlotItems.Data(((min(x),0),(max(x),0)), with_="lines linetype 2 linewidth 2")
secondPline = Gnuplot.PlotItems.Data(((0,min(y)),(0,max(y))), with_="lines linetype 3 linewidth 2")
gp.plot(realDataplot, firstPline, secondPline)
    
raw_input('Press the Return key to quit: ')

