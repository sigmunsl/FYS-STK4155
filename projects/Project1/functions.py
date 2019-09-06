## Functions
import numpy as np


def FrankeFunction(x, y, noise_level):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_level*np.random.randn(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise

def OridinaryLeastSquares(design, data):
    inverse_term   = np.inv(design.T.dot(design))
    beta           = inverse_term.dot(design.T).dot(data)
    return beta


def RidgeRegression(design, data, _lambda):
    inverse_term   = np.inv(design.T.dot(design)+ _lambda*np.eye((design.shape[1]))
    beta           = inverse_term.dot(design.T).dot(data)
    return beta

def VarianceBeta(design, _lambda):
    vb = np.linalg.inv(design.T.dot(design) + _lambda*np.eye((design.shape[1])))
    return np.diag(vb)

def MSE(y,ytilde):
    MeanSquaredError = (np.sum((y-ytilde)**2))/len(y)

def R2Score(y,ytilde):
    mean_value   = (np.sum(y))/len(y)
    numerator    = (np.sum((y-ytilde)**2))
    denomenator  = (np.sum((y-mean_value)**2))
    R2           = (1-(numerator/denomenator))
    return R2  
