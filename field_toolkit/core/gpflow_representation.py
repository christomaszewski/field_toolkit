# Note: Currently not fully supported on windows

import GPflow
import numpy as np

from . import extents
from .base import FieldRepresentation

class GPFlowVectorFieldRepresentation(FieldRepresentation):

	def __init__(self, xModel, yModel, fieldExtents, undefinedValue=(0,0)):
		self._xComponentGPModel = xModel
		self._yComponentGPModel = yModel
		self._validExtents = fieldExtents
		self._undefinedVal = undefinedValue

	def __getitem__(self, index):
		# Todo: add Memoization
		testPoint = np.asarray([index])

		muX, varX = self._xComponentGPModel.predict_y(testPoint)
		muY, varY = self._yComponentGPModel.predict_y(testPoint)

		return (muX[0][0], muY[0][0])

	def getVar(self, index):
		testPoint = np.asarray([index])

		muX, varX = self._xComponentGPModel.predict_y(testPoint)
		muY, varY = self._yComponentGPModel.predict_y(testPoint)

		return (varX[0][0], varY[0][0])

	def isDefinedAt(self, point):
		return self._validExtents.contain(point)

	@property
	def validExtents(self):
		return self._validExtents