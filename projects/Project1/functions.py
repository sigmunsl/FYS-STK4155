## Functions
import numpy as np


def FrankeFunction(x, y, noise_level=0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_level*np.random.randn(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise

def OridinaryLeastSquares(design, data, test):
    inverse_term   = np.linalg.inv(design.T.dot(design))
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred

def OridinaryLeastSquares_SVD(design, data, test):
    U,_sigma,V     = np.linalg.svd(design.T.dot(design))
    inverse_term   = 23
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred


def RidgeRegression(design, data, test, _lambda=0):
    inverse_term   = np.linalg.inv(design.T.dot(design)+ _lambda*np.eye((design.shape[1])))
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred

def VarianceBeta(design, _lambda=0):
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

def CreateDesignMatrix_X_morten(x, y, n = 5):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X

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

    X,Y          = np.meshgrid(x,y)
    X            = np.ravel(X)
    Y            = np.ravel(Y)
    DesignMatrix = np.empty((len(X),len(concat_x)))
    for i in range(len(concat_x)):
        DesignMatrix[:,i]   = (X**concat_x[i])*(Y**concat_y[i])

    DesignMatrix = np.concatenate((np.ones((len(X),1)),DesignMatrix), axis = 1)
    return DesignMatrix


def k_fold_cv(k, data, design, predictor, _lambda=0):
    data = np.reshape(k, int(len(data)/k))
    design = np.reshape(k, int(len(design[0, :])/k), len(design[:, 0]))
    r2 = 0
    mse = 0
    for i in range(k):
        if _lambda != 0:
            beta, pred = predictor(design[np.arange(len(design))!=i], data[np.arange(len(data))!=i], design[i], _lambda)
        else:
            beta, pred = predictor(design[np.arange(len(design))!=i], data[np.arange(len(data))!=i], design[i])
        r2 += R2Score(data[i], pred)
        mse += MSE(data[i], pred)
    return r2/k, mse/k
