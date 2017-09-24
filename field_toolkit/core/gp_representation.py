import GPy
import numpy as np

from . import extents
from .representation import VectorFieldRepresentation, ScalarFieldRepresentation

# Todo: Add GP Scalar Field Representation

class GPVectorFieldRepresentation(VectorFieldRepresentation):

	def __init__(self, xModel, yModel, fieldExtents, undefinedValue=(0.,0.)):
		self._xComponentGPModel = xModel
		self._yComponentGPModel = yModel
		self._validExtents = fieldExtents
		self._undefinedVal = undefinedValue

	def __getitem__(self, index):
		testPoint = np.asarray([index])

		muX, varX = self._xComponentGPModel.predict_noiseless(testPoint)
		muY, varY = self._yComponentGPModel.predict_noiseless(testPoint)

		return (muX[0][0], muY[0][0])

	def getVar(self, index):
		testPoint = np.asarray([index])

		muX, varX = self._xComponentGPModel.predict_noiseless(testPoint)
		muY, varY = self._yComponentGPModel.predict_noiseless(testPoint)

		return (varX[0][0], varY[0][0])


class CoregionalizedGPFieldRepresentation(VectorFieldRepresentation):

	def __init__(self, model, fieldExtents, undefinedValue=(0,0)):
		self._gpModel = model
		self._validExtents = fieldExtents
		self._undefinedVal = undefinedValue

	def __getitem__(self, index):
		x, y = index
		newX = np.asarray([[x]])
		newY = np.asarray([[y]])
		newInputX = np.hstack([newX, newY, np.zeros_like(newX)])
		newInputY = np.hstack([newX, newY, np.ones_like(newX)])
		
		muX, varX = self._gpModel.predict_noiseless(newInputX)
		muY, varY = self._gpModel.predict_noiseless(newInputY)
		
		return (muX[0][0], muY[0][0])

	def getVar(self, index):
		x, y = index
		newX = np.asarray([[x]])
		newY = np.asarray([[y]])
		newInputX = np.hstack([newX, newY, np.zeros_like(newX)])
		newInputY = np.hstack([newX, newY, np.ones_like(newX)])
		
		muX, varX = self._gpModel.predict_noiseless(newInputX)
		muY, varY = self._gpModel.predict_noiseless(newInputY)
		
		return (varX[0][0], varY[0][0])


class GPScalarFieldRepresentation(ScalarFieldRepresentation):

	def __init__(self, model, fieldExtents, undefinedValue=(0,0)):
		self._gpModel = model
		self._validExtents = fieldExtents
		self._undefinedVal = undefinedValue

	def __getitem__(self, index):
		testPoint = np.asarray([index])

		mu, var = self._gpModel.predict_noiseless(testPoint)

		return mu[0][0]

	def getVar(self, index):
		testPoint = np.asarray([index])

		mu, var = self._gpModel.predict_noiseless(testPoint)

		return var[0][0]