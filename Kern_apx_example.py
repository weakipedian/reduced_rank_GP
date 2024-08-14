import numpy as np
import yaml
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

# Generate 2D training data
X = np.linspace(-5, 5, 20)
Y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(X, Y)
Z = complexFunction(X, Y) + np.random.normal(0, config['sensorNoise'], X.shape)

# Transform the data into a [N x 2] matrix
XTrain = np.vstack([X.ravel(), Y.ravel()]).T
yTrain = Z.ravel()

# Generate a test grid for prediction
xPred = np.linspace(-5, 5, 50)
yPred = np.linspace(-5, 5, 50)
XPred, YPred = np.meshgrid(xPred, yPred)
XYPred2D = np.vstack([XPred.ravel(), YPred.ravel()]).T

# Original Kernel
kernel = GPy.kern.Matern32(input_dim=2, variance=1.0, lengthscale=1.0)
K = kernel.K(XYPred2D, np.array([[0, 0]]))

# Reduced Rank Kernel
SparseGP = ReducedRankGP(config)
predPhi = SparseGP.buildEigenfunctions(XYPred2D)
originPhi = SparseGP.buildEigenfunctions(np.array([[0, 0]]))
Kapprox = np.dot(np.dot(predPhi.T, SparseGP.spectDensity), originPhi)

# RMSE
rmse = np.sqrt(np.mean((K - Kapprox)**2))
print("RMSE: ", rmse)

# Visualization
fig = plt.figure(figsize=(12, 6))
kernGP = fig.add_subplot(121, projection='3d')
kernGP.plot_surface(XPred, YPred, K.reshape(50, 50), cmap='viridis', antialiased=False)
kernGP.set_zlim(0, 1)
kernGP.set_xlabel('X')
kernGP.set_ylabel('Y')
kernGP.set_zlabel('Kernel Value')
plt.title('Matern 3/2 Kernel Output (2D Input)')

apxGP = fig.add_subplot(122, projection='3d')
apxGP.plot_surface(XPred, YPred, Kapprox.reshape(50, 50), cmap='viridis', antialiased=False)
apxGP.set_zlim(0, 1)
apxGP.set_xlabel('X')
apxGP.set_ylabel('Y')
apxGP.set_zlabel('Kernel Value')

plt.title('Matern 3/2 Kernel Output (2D Input)')
plt.show()
