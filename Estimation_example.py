import numpy as np
import yaml, time
import GPy
import matplotlib.pyplot as plt
from ReducedRankGPLib import ReducedRankGP

# Complex 2D function: a function with multiple frequency components
def complexFunction(x, y):
    return np.sin(x) * np.cos(y) + 0.5 * np.sin(2 * x) * np.cos(2 * y)

# Should be replaced with the path to your config file
config_file_path = '/root/catkin_ws/src/gp_mapping/scripts/config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

kernParams = config['kernParams']

# Generate 2D training data
trainX = np.linspace(-5, 5, 50)
trainY = np.linspace(-5, 5, 50)
trainX, trainY = np.meshgrid(trainX, trainY)
trainZ = complexFunction(trainX, trainY) + np.random.normal(0, config['sensorNoise'], trainX.shape)

# Transform the data into a [N x 2] matrix
trainXY = np.vstack([trainX.ravel(), trainY.ravel()]).T
trainZ = trainZ.ravel().reshape(-1, 1)

# Generate a test grid for prediction
testX = np.linspace(-5, 5, 100)
testY = np.linspace(-5, 5, 100)
testX, testY = np.meshgrid(testX, testY)
testXY = np.vstack([testX.ravel(), testY.ravel()]).T

########### Kernel-based GP ###########
kernel = GPy.kern.Matern32(input_dim=config['inputDim'], variance=1.0, lengthscale=kernParams['l'])
startKGPupdate = time.time()
KGPmodel = GPy.models.GPRegression(trainXY, trainZ, kernel, noise_var=config['sensorNoise'])
endKGPupdate = time.time()

startKGPpredict = time.time()
predKernGP, varKernGP = KGPmodel.predict(testXY)
endKGPpredict = time.time()

########### Reduced-rank GP ###########
SGPmodel = ReducedRankGP(config)
startSGPupdate = time.time()
SGPmodel.modelUpdate(trainXY, trainZ)
endSGPupdate = time.time()

startKGPpredict = time.time()
predSGP, varSGP = SGPmodel.predict(testXY)
endKGPpredict = time.time()

# Visualization
fig = plt.figure(figsize=(20, 6))
fig.suptitle(f'Comparison of Ground Truth, Kernel-based GP, and Reduced-rank GP with measurements (N={trainXY.shape[0]})')

groundTruth = fig.add_subplot(131)
groundTruth.contourf(testX, testY, complexFunction(testX, testY), levels=100, cmap='viridis')
groundTruth.set_xlabel('X')
groundTruth.set_ylabel('Y')
groundTruth.set_title('[Ground Truth]')

kernGP = fig.add_subplot(132)
kernGP.contourf(testX, testY, predKernGP.reshape(testX.shape), levels=100, cmap='viridis')
kernGP.set_xlabel('X')
kernGP.set_ylabel('Y')
kernGP.set_title(f'[Kernel-based GP], update time: {endKGPupdate - startKGPupdate:.2f}s, predict time: {endKGPpredict - startKGPpredict:.2f}s')

apxGP = fig.add_subplot(133)
apxGP.contourf(testX, testY, predSGP.reshape(testX.shape), levels=100, cmap='viridis')
apxGP.set_xlabel('X')
apxGP.set_ylabel('Y')
apxGP.set_title(f'[Reduced-rank GP], update time: {endSGPupdate - startSGPupdate:.2f}s, predict time: {endKGPpredict - startKGPpredict:.2f}s')

plt.show()
