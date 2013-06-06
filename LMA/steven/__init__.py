# -*- coding:utf-8 -*-

# @author: StevenLee @contact: StuNo.11221078 Phone:13646823639

import Gnuplot, Gnuplot.funcutils
import numpy as np
from numpy import *

def gradient(func, dims, params, delta):
    
    grad = np.zeros(dims)
    tmp = np.zeros(dims)

    # Compute the gradient
    for i in range(dims):
        tmp[i] = delta
        grad[i] = (func(*(params + tmp)) - func(*(params))) / delta
        tmp[i] = 0
        
    return grad

def hessian(func, dims, params, delta):
   
    hessian = np.zeros((dims, dims))
    tmp_i = np.zeros(dims)
    tmp_j = np.zeros(dims)

    for i in xrange(dims):
    
        tmp_i[i] = delta
        params_1 = params + tmp_i
        params_2 = params - tmp_i
        
        for j in xrange(i, dims):
        
            tmp_j[j] = delta
            deriv_2 = (func(*(params_2 + tmp_j)) - func(*(params_1 + tmp_j))) / delta
            deriv_1 = (func(*(params_2 - tmp_j)) - func(*(params_1 - tmp_j))) / delta
            hessian[i][j] = (deriv_2 - deriv_1) / delta
            
            hessian[j][i] = hessian[i][j] # Since the Hessian is symmetric, spare me some calculations
            
            tmp_j[j] = 0
        
        tmp_i[i] = 0
    
    return hessian

def leven_marquardt(func, dims, initial, maxit=100, stop=1e-15, delta = 0.0001):
    
    solution = np.copy(initial)
    estimates = [solution]
    goals = [func(*solution)]

    #print goals

    mu = delta * 10
    mus = [mu]
    qs = goals
    
    for i in range(maxit):
        
        grad = gradient(func, dims, solution, delta)
        H = hessian(func, dims, solution, delta)

        if (grad ** 2).sum() < stop:
            break;

        is_singular = True
        while is_singular:
            try:
                correction = np.linalg.solve(H + mu * np.eye(H.shape[0]), grad)
                is_singular = False
            except Exception, e:
                mu = 4 * mu
                mus.append(mu)

        solution = solution + correction
        estimates.append(solution)
        goals.append(func(*solution))

        q = goals[i + 1] + (grad * correction).sum() + 0.5 * np.mat(correction) * np.mat(H) * np.mat(correction).T
        qs.append(q)
        r = (goals[i + 1] - goals[i]) / (qs[i + 1] - qs[i])
        
        if r <= 0:
            # solution remains unchanged
            solution = solution - correction
        if r < 0.25:
            mu = 4 * mu
            mus.append(mu)
        elif r > 0.75:
            mu = 0.5 * mu
            mus.append(mu)
        else :
            pass
            
    return solution, mus, goals[-1], np.array(estimates), goals

def object_functions(x):
    return x**2
    #x**4 + (x-1)**3 + (x-3)**2 + 4


print '*********Leven-Marquardt Method*********'

print 'Step1: LMA Process----------------------'
solution, mus, result, estimates, goals = leven_marquardt(object_functions, 1, [70])

print "solution: "+ str(solution)
print "result: ",result
print "estimates: "+ str(estimates)
print "goals: "+ str(goals)

print 'Step2: Plot the result----------------------'
gp = Gnuplot.Gnuplot() # define a GNUplot object

# ----  set the parameters
gp('set data style lines')
gp.title("Leven Marquardt Algorithm @author: StevenLee")
gp.xlabel('x')
gp.ylabel('y')

estimatesarray = [estimates[i,0] for i in range(estimates.shape[0])]
estimatesarray[::-1] #inverted order
#print estimatesarray

estimatesData = []
for item in estimatesarray:
    estimatesData.append((item,object_functions(item)))

t = np.arange(-200.0, 200.0, 0.02)
realDataplot = Gnuplot.PlotItems.Data(t,object_functions(t), with_="lines linetype 3 linewidth 1", title=" x**2")
estimatesDataplot = Gnuplot.PlotItems.Data(estimatesData,with_="lines linetype 1 linewidth 3", title="estimatesData")

gp.plot(realDataplot,estimatesDataplot)

raw_input('Press the Return key to quit: ')
