## Functions
import numpy as np


def FrankeFunction(x, y, noise_level=0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_level*np.random.randn(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise

def OridinaryLeastSquares(design, data):
    inverse_term   = np.linalg.inv(design.T.dot(design))
    beta           = inverse_term.dot(design.T).dot(data)
    return beta


def RidgeRegression(design, data, _lambda):
    inverse_term   = np.linalg.inv(design.T.dot(design)+ _lambda*np.eye((design.shape[1])))
    beta           = inverse_term.dot(design.T).dot(data)
    return beta

def VarianceBeta(design, _lambda):
    vb = np.linalg.inv(design.T.dot(design) + _lambda*np.eye((design.shape[1])))
    return np.diag(vb)

def MSE(y, ytilde):
    MeanSquaredError = (np.sum((y-ytilde)**2))/y.size
    return MeanSquaredError

def R2Score(y, ytilde):
    mean_value   = (np.sum(y))/y.size
    numerator    = (np.sum((y-ytilde)**2))
    denomenator  = (np.sum((y-mean_value)**2))
    R2           = (1-(numerator/denomenator))
    return R2



def DesignDesign(x, y, power):
    concat_x   = np.array([0,0])
    concat_y   = np.array([0,0])

    for i in range(power):
        toconcat_x = np.arange(i+1,-1,-1)
        toconcat_y = np.arange(0,i+2,1)
        concat_x   = np.concatenate((concat_x,toconcat_x))
        concat_y   = np.concatenate((concat_y,toconcat_y))

    concat_x     = concat_x[2:len(concat_x)]
    concat_y     = concat_y[2:len(concat_y)]
    DesignMatrix = np.empty((len(x),len(concat_x)))

    for i in range(len(concat_x)):
        DesignMatrix[:,i]   = (x**concat_x[i])*(y**concat_y[i])

    return DesignMatrix
