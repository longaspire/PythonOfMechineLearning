# -*- coding:utf-8 -*-

# @author: StevenLee @contact: StuNo.11221078 Phone:13646823639

from numpy import *
import numpy as np
import Gnuplot, Gnuplot.funcutils

k = 2  # the number of clusters
threshold = 1e-15  # threshold
Lprev = inf # the previous L
maxStep = 100  # max step in EM 
m = 500  # the number of samples
samples = []

def Generate2DGaussian(sampleNum):
    x = (float)(random.uniform(-10000, 10000)) / 1000
    y = (float)(random.uniform(-10000, 10000)) / 1000
    # print x,y
    m_mean = (x, y)
    m_cov = [[(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000], [(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000]]
    
    m_sample = np.random.multivariate_normal(m_mean, m_cov, sampleNum)
    return m_sample
    
def GenerateMaxtureGaussian(sampleNum):
    
    frac = 0.5  # (float)(random.uniform(1, 10000))/10000
    # print frac
    num_first = (int)(frac * sampleNum)
    num_second = sampleNum - num_first
    #print num_first, num_second
    samples_first_Gaussian = Generate2DGaussian(num_first)  
    samples_second_Gaussian = Generate2DGaussian(num_second)
    
    m_samples = np.vstack((samples_first_Gaussian, samples_second_Gaussian))
    
    return m_samples

def calMixGaussian(x, miu, sigma, n):
    numerator = np.exp(-0.5 * np.matrix(x - miu).T * np.linalg.inv(sigma) * np.matrix(x - miu))
    denominator = np.power(2 * np.pi, n / 2) * np.sqrt(np.abs(np.linalg.det(sigma)))
    '''numerator is a matrix'''
    px = numerator[(0, 0)] / denominator
    return px

def init_params(sampleNum):
    probability = []
    
    for i in range(sampleNum):
        x = random.random()
        # print x
        probability.append(np.array([x, 1 - x]))
        
    print shape(probability)
    Cov, Mean = [], []
    
    for i in range(2):
        Cov.append(np.array([(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000]))
        Mean.append([[(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000], [(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000]])
        
    print shape(Cov)
    print shape(Mean)
    
    x = random.random()
    Pi = np.array([x, 1 - x])
    print shape(Pi)


def EM(sampleNum):
    
    probability = []
    L,Lprev = 0.0, 0.0
    for i in range(sampleNum):
        x = random.random()
        # print x
        probability.append(np.array([x, 1 - x]))
        
    print shape(probability)
    Cov, Mean = [], []
    for i in range(2):
        Mean.append(np.array([(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000]))
        Cov.append([[(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000], [(float)(random.uniform(-10000, 10000)) / 1000, (float)(random.uniform(-10000, 10000)) / 1000]])
        
    print shape(Cov)
    print shape(Mean)
    
    x = random.random()
    Pi = np.array([x, 1 - x])
    print shape(Pi)
    
    step = 0
    while(step < maxStep):
        
        # E-step
        for i in range(sampleNum):
            # print samples[i]
            px1 = calMixGaussian(np.matrix(samples[i]).T, np.matrix(Mean[0]).T, np.matrix(Cov[0]), 2)
            px2 = calMixGaussian(np.matrix(samples[i]).T, np.matrix(Mean[1]).T, np.matrix(Cov[1]), 2)  
            probability[i][0] = Pi[0] * px1 / (Pi[0] * px1 + Pi[1] * px2)
            probability[i][1] = Pi[1] * px2 / (Pi[0] * px1 + Pi[1] * px2)
            L += (Pi[0] * px1 + Pi[1] * px2)
        #print L
            
        # M-step
        for j in range(k):
            Pi[j] = 0
            sumOfProb = 0.0
            for i in range(m):
                sumOfProb += probability[i][j]
            Pi[j] = sumOfProb / m
            
        for j in range(k):
            weight = 0
            sumOfProb = 0.0
            for i in range(m):
                weight += samples[i] * probability[i][j]
                sumOfProb += probability[i][j]
            Mean[j] = weight / sumOfProb
                
        for j in range(k):
            weight = 0
            sumOfProb = 0.0
            for i in range(m):
                weight += np.matrix(samples[i] - Mean[j]).T * np.matrix(samples[i] - Mean[j]) * probability[i][j]
                sumOfProb += probability[i][j]
            Cov[j] = weight / sumOfProb
        
        
        #if (L-Lprev) < threshold:
            #break
        
        Lprev = L
          
        step = step + 1
        print 'step:', step
    
    samples_firstGaussian = []
    samples_secondGaussian = []
    for i in range(m):
        if probability[i][0] > 0.5:
            samples_firstGaussian.append(samples[i])
        else:
            samples_secondGaussian.append(samples[i])
    
    print Mean[0]
    print Mean[1]
    print Cov[0]
    print Cov[1]
    
    return samples_firstGaussian, samples_secondGaussian
    

print 'GMM solving----------------------------'

print 'Generate the GMM------------------------'
samples = GenerateMaxtureGaussian(m)
# print samples[999]

print 'Expectation Maximization------------------------'
samples_firstGaussian, samples_secondGaussian = EM(m)

print 'Plot all the points------------------------'
gp = Gnuplot.Gnuplot()  # define a GNUplot object
gp('set data style lines')
gp.title("Gaussian Mixture Model @author: StevenLee")
gp.xlabel('x')
gp.ylabel('y')
 
Dataplot_1 = Gnuplot.PlotItems.Data(samples_firstGaussian, with_="points pointtype 7 pointsize 0")
Dataplot_2 = Gnuplot.PlotItems.Data(samples_secondGaussian, with_="points pointtype 5 pointsize 0")
gp.plot(Dataplot_1, Dataplot_2)
    
raw_input('Press the Return key to quit: ')
