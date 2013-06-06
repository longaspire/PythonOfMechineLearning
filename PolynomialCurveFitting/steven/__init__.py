# -*- coding:utf-8 -*-

# @author: StevenLee @contact: StuNo.11221078 Phone:13646823639

import Gnuplot, Gnuplot.funcutils
import math
from scipy.optimize import leastsq
from numpy import *

x_val = []
y_val_real = []
y_val_random = []

def getRandomValuesFromRealData(Data, bias, count):
    # ----  as a storage of random values 
    randomData = []
    #print bias
    for i in range(0,101,100/(count-1)):
        print Data[i]
        incremental = random.uniform(-1,1)*bias + Data[i][1]
        y_val_random.append(incremental)
        randomData.append((Data[i][0], incremental))
    return randomData

def printRandomData(Data):
    for DataItem in Data:
        print DataItem
        
def Regression(Data, count):
    
    Vector_X = []
    Vector_X_Item = []
    Vector_Y = []
    Vector_Y_Item = []
    
    for DataItem in Data:
        Vector_X_Item = []
        for i in range(0, count):
            Vector_X_Item.append(math.pow(DataItem[0], i))
        Vector_Y_Item.append(DataItem[1])
        Vector_X.append(Vector_X_Item)
    Vector_Y.append(Vector_Y_Item)
    
    #print Vector_X
    #print Vector_Y
    matrix_X = array(Vector_X)
    matrix_Y = array(Vector_Y)
    matrix_X_T = matrix_X.T
    matrix_dot_XandXT_I = matrix(dot(matrix_X_T, matrix_X)).I
    matrix_dot_front = dot(matrix_dot_XandXT_I, matrix_X_T).T
    matrix_W = dot(matrix_Y, matrix_dot_front)
    #print matrix_dot_front.shape
    #print matrix_Y.shape
    
    print matrix_W
    Vector_W = matrix_W.T
    print Vector_W
    #print matrix_X
    #print matrix_X_T
    #print dot(matrix_X, matrix_X_T)
    #print matrix_Y_T,matrix_Y
    #print dot(matrix_dot_XandXT_I, matrix_X_T)
    return Vector_W

def produceFittingCurve(Vector, count):
    RegressionData = []
    for v in range(0,101,1):
        x = (v*1.0)/100
        t = 0
        for i in range(0,count):
            t += Vector[i][0]*pow(x,i)
            #RegressionDataItem = []
            RegressionData.append((x,t))
        #print "v = %d   t = %f" %(v, t)
    return RegressionData

def produceFittingCurveWithPenalty(xval, Vector, count):
    RegressionDataWithPenalty = []
    for v in xval:
        t = 0
        for i in range(0,count):
            t += Vector[i]*pow(v,i)
            #RegressionDataItem = []
        RegressionDataWithPenalty.append((v,t))
        #print "v = %d   t = %f" %(v, t)
    return RegressionDataWithPenalty

def PolyFunction(x, w, degree):
    val = 0
    for i in range(0,degree+1):
        val += w[i] * pow(x,i)
    return val

def residuals_PolyFunction_simple(p, deg, y, x):
    err = y-PolyFunction(x, p, deg)
    return err

def residuals_PolyFunction_multiple(p, deg, y, x, lambda_term):
    err = y - PolyFunction(x, p, deg)
    err = append(err, sqrt(lambda_term) * p)
    return err


print '*********Solving Polynomial Curve Fitting Problem*********'
#print math.sin(math.pi/2)

gp = Gnuplot.Gnuplot() # define a GNUplot object

# ----  set the parameters
gp('set data style lines')
gp.title("Polynomial Curve Fitting @author: StevenLee")
gp.xlabel('x')
gp.ylabel('t')
gp.set_range("xrange", (0,1))
gp.set_range("yrange", (-1.5,1.5))

# ----  initialize the parameters
count_N = 100      # count of the sample points n(x) [which n should be ≥ 3]
count_W = 9        # count of the weights W
value_lambda = e**(-18)   # value of the penalize parameter lambda

# ----  as a storage of real curve
realData = []
realDataItem = []

print 'Step1: produce the sin(x) real points----------------------'
for v in range(0,101,1):
    x = (v*1.0)/100
    t = math.sin(x*math.pi*2)
    #print "v = %d   t = %f" %(v, t)
    realDataItem = []
    realData.append((x,t))
    x_val.append(x)
    y_val_real.append(t)

print 'Step2: get Random Values From the Real Data----------------------'
randomData = getRandomValuesFromRealData(realData, 0.2, count_N)

print 'Step3: print the Random Values From the Real Data----------------------'
printRandomData(randomData)

print 'Step4: regression process get the weights matrix----------------------'
Vector_W = Regression(randomData, count_W+1)
#print Vector_W[1][0]

print 'Step5: produce the fitting curve----------------------'
RegressionData = []
RegressionData = produceFittingCurve(Vector_W, count_W+1)

print 'Step6: produce the fitting curve with penalty----------------------'
factor_w = range(1, count_W+2, 1)
#print factor_w
x_val = linspace(0, 1.0, count_N+1)

polysq = leastsq(residuals_PolyFunction_multiple, factor_w, args=(count_W, y_val_random, x_val, value_lambda))
print polysq[0]
RegressionDataWithPenalty = []
RegressionDataWithPenalty = produceFittingCurveWithPenalty(x_val, polysq[0], count_W+1)


realDataplot = Gnuplot.PlotItems.Data(realData, with_="lines linetype 5 linewidth 1", title="y=sin(x)")
randomDataplot = Gnuplot.PlotItems.Data(randomData, with_="points pointtype 7 pointsize 1", title="random points")
regressionDataplot = Gnuplot.PlotItems.Data(RegressionData, with_="lines linetype 2 linewidth 2", title="Polynomial_Curve_Fitting")
regressionDataWithPenaltyplot = Gnuplot.PlotItems.Data(RegressionDataWithPenalty, with_="lines linetype 12 linewidth 2", title="Polynomial_Curve_Fitting_With_Penalty")

gp.plot(randomDataplot, realDataplot, regressionDataplot, regressionDataWithPenaltyplot)

raw_input('Press the Return key to quit: ')

