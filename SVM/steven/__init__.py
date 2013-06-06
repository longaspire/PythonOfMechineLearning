# -*- coding:utf-8 -*-

# @author: StevenLee @contact: StuNo.11221078 Phone:13646823639

import Gnuplot, Gnuplot.funcutils
import numpy as np
from numpy import *
import cvxopt
import cvxopt.solvers
import pylab as pl

def gen_lin_separable_data(Num):
    
    mean_1 = np.array([0, 2])
    mean_2 = np.array([2, 0])
    cov = np.array([[0.6, 0.8], [0.8, 0.6]])
    X1 = np.random.multivariate_normal(mean_1, cov, Num)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean_2, cov, Num)
    y2 = np.ones(len(X2)) * -1
        
    return X1, y1, X2, y2
    
def split_trainingData_and_testData(X1, y1, X2, y2, Num):
    
    frac = Num * 0.9
    
    X1_train = X1[:frac]
    y1_train = y1[:frac]
    X2_train = X2[:frac]
    y2_train = y2[:frac]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    
    X1_test = X1[frac:]
    y1_test = y1[frac:]
    X2_test = X2[frac:]
    y2_test = y2[frac:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    
    return X_train, y_train, X_test, y_test

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.a), n_samples)

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

def plot_result(X1_train, X2_train, clf, gp):
    
    X1, X2 = np.meshgrid(np.linspace(-2,2,50), np.linspace(-2,2,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    
    for i in range(0,size(Z[:,0])):
        print i, Z[i,0]/Z[i,1]
    
    realDataplot_1 = Gnuplot.PlotItems.Data(X1_train[:,0], X1_train[:,1], with_="points pointtype 7 pointsize 0", title="Set1")
    realDataplot_2 = Gnuplot.PlotItems.Data(X2_train[:,0], X2_train[:,1], with_="points pointtype 5 pointsize 0", title="Set2")
    realDataplot_3 = Gnuplot.PlotItems.Data(clf.sv[:,0], clf.sv[:,1], with_="points pointtype 6 pointsize 0", title="SV")
    realDataplot_4 = Gnuplot.PlotItems.Data(Z[:,0] - 1, Z[:,1] - 1, with_="lines linetype 3 linewidth 2")
    realDataplot_5 = Gnuplot.PlotItems.Data(Z[:,0] - 2, Z[:,1] - 1, with_="lines linetype 1 linewidth 2")
    realDataplot_6 = Gnuplot.PlotItems.Data(Z[:,0], Z[:,1] - 1, with_="lines linetype 2 linewidth 2")
    
    gp.plot(realDataplot_1,realDataplot_2,realDataplot_3,realDataplot_4,realDataplot_5,realDataplot_6)
    
    
        
print '*********Supported Vector Machine*********'

PointsNum = 200

print 'Step1: Generate the data for separation----------------------'
X1, y1, X2, y2 = gen_lin_separable_data(PointsNum)
#print X1, y1, X2, y2

print 'Step2: Split training Data and test Data----------------------'
X_train, y_train, X_test, y_test = split_trainingData_and_testData(X1, y1, X2, y2, PointsNum)

print 'Step1: SVM Process----------------------'
mySVM = SVM(C=0.1)
mySVM.fit(X_train, y_train)

y_predict = mySVM.predict(X_test)
correct = np.sum(y_predict == y_test)
print "%d out of %d predictions correct" % (correct, len(y_predict))


print 'Step2: Plot the result----------------------'
gp = Gnuplot.Gnuplot() # define a GNUplot object

# ----  set the parameters
gp('set data style lines')
gp.title("Supported Vector Machine @author: StevenLee")
gp.xlabel('x')
gp.ylabel('y')

plot_result(X_train[y_train==1], X_train[y_train==-1], mySVM, gp)

raw_input('Press the Return key to quit: ')
