#!/usr/bin/python3

import numpy as np
from scipy.special import kv, gamma
from scipy import linalg

class ReducedRankGP:
    def __init__(self, config):
      self.boundaryL = config['boundaryL']
      self.numE = config['numE'] # the number of elements of E
      self.sensorNoise = config['sensorNoise']
      self.inputDim = config['inputDim']        
      
      self.trainInputMat = np.zeros((self.numE, self.numE))
      self.trainOutputVec = np.zeros((self.numE, 1))
      
      ########### Compute eigenvalues and init eigenfunctions ###########
      # Spectral density matrix of matern kernel could be initialized in advance
      self.buildSpectDensityMat(config['covFunc'], config['kernParams'])

    def modelUpdate(self, trainXY, trainZ):
      """
      inputs: trainXY (numTrain, inputDim)
              trainZ (numTrain, 1)
              predXY (numPred, inputDim)
      outputs: predZ (numPred, 1)
      """
      # Compute eigenfunctions
      trainPhi = self.buildEigenfunctions(trainXY) # (numE, numTrain)
      self.trainInputMat += np.dot(trainPhi, trainPhi.T) # (numE, numE)
      self.trainOutputVec += np.dot(trainPhi, trainZ) # (numE, 1)
      self.midTerm = linalg.inv(self.trainInputMat + self.sensorNoise * self.inverseSpectDensity)
    def predict(self, testXY):
      """
      inputs: testXY (numTest, inputDim)
      outputs: predZ (numTest, 1)
      """
      predPhi = self.buildEigenfunctions(testXY)
      
      mean = np.dot(predPhi.T, np.dot(self.midTerm, self.trainOutputVec)) # (numTest, 1)
      var = np.diag( np.dot(np.dot(predPhi.T, self.inverseSpectDensity), predPhi) - np.dot(np.dot(predPhi.T, self.inverseSpectDensity), np.dot(self.trainInputMat, np.dot(self.inverseSpectDensity, predPhi))) ) # (numTest, 1)

      return mean, var
    
    def buildSpectDensityMat(self, conFunc, kernParams):
      index = np.arange(0,80) # the number of elements of E
      index = index.astype(np.uint64)
      index = index.reshape(-1,1)

      ########### Compute eigenvalues using Hilbert space method ###########
      eigenValues = (np.pi * (index + 1) / (2*self.boundaryL))**2
      eigenValues = eigenValues.reshape(-1,1) # (numE, 1)
      frequency = np.sqrt(eigenValues) # (numE, 1)
      
      # Combination of integer orders 
      sortedFreq, self.sortedOrder = self.sortFrequency(frequency, index) # (numE, 1) , (numE, inputDim)
      spectDensity = self.SpectralDensityFunc(conFunc, kernParams, sortedFreq) # (numE, 1)
      self.spectDensity = np.diag(spectDensity[:,0]) # (numE, numE)
      self.inverseSpectDensity = linalg.inv(self.spectDensity) # (numE, numE)

    def buildEigenfunctions(self, xy):
      """
      inputs: xy ( len(xy) , inputDim)
      outputs: phi (numE, len(xy))
      """
      if self.inputDim == 1:
        pass
      elif self.inputDim == 2:
        ret = np.zeros((self.numE, xy.shape[0]))
        idx = 0
        for i,j in self.sortedOrder:
            ret[idx, :] = (1/np.sqrt(self.boundaryL))**self.inputDim * np.sin( np.pi * (i+1) * ( xy[:,0] + self.boundaryL ) / (2*self.boundaryL) ) * np.sin( np.pi * (j+1) * ( xy[:,1] + self.boundaryL ) / (2*self.boundaryL) )
            idx += 1
      elif self.inputDim == 3:
        pass
      
      return ret
    
    
    def SpectralDensityFunc(self, covFunc, kernParams, frequency):
      """
      inputs: frequency (numE, 1)
      outputs: spectral density vector (numE, 1)
      """
      kernType = covFunc['kernType']
      if kernType == 'Matern':
        nominator = 2**self.inputDim * np.pi**(self.inputDim/2) * gamma(kernParams['v'] + self.inputDim/2) * (2 * kernParams['v'])**(kernParams['v'])
        denominator = gamma(kernParams['v']) * kernParams['l']**(2*kernParams['v']) * (2*kernParams['v']/kernParams['l']**2 + frequency**2)**( kernParams['v'] + self.inputDim/2 )
        ret = nominator/denominator
      elif kernType == 'SE':
        pass
      
      return ret
    
    def sortFrequency(self, frequency, index):
      """
      inputs: frequency (numE, 1)
      outputs: sorted frequency vector (numE, 1)
               sorted index matrix (numE, inputDim)
      """
      if self.inputDim == 1:
        pass
      elif self.inputDim == 2:
        # (i, j)
        xIndex, yIndex = np.meshgrid(index, index)
        xyIndices = np.vstack([xIndex.ravel(), yIndex.ravel()]).T
        
        # (freq_i, freq_j)
        xFreq, yFreq = np.meshgrid(frequency, frequency)
        xyFreq = np.vstack([xFreq.ravel(), yFreq.ravel()]).T
        
        # Sorting the xyFreq
        normFreq = np.linalg.norm(xyFreq, axis=1)
        tempMat = np.append(normFreq.reshape(-1, 1), xyIndices, axis=1)
        
        sortedMat = tempMat[np.lexsort(np.fliplr(tempMat).T)]
        sortedOrder = sortedMat[:self.numE,1:].astype(np.uint64)
        sortedFreq = sortedMat[:self.numE,0].reshape(-1, 1)
        
      elif self.inputDim == 3:
        pass
      return sortedFreq, sortedOrder    
      
