import GPflow
import numpy as np

from ..core import vf
from .base import VectorFieldApproximator

class GPFlowApproximator(VectorFieldApproximator):

	def __init__(self, kernel=None):
		self._measurements = []

		if (kernel is None):
			# Default kernel
			self._Kx = GPflow.kernels.Matern52(input_dim=2, ARD=True)
			#			GPy.kern.RatQuad(input_dim=2, ARD=True) +\
			#			GPy.kern.White(input_dim=2)
			self._Ky = GPflow.kernels.Matern52(input_dim=2, ARD=True)
		else:
			self._Kx = kernel
			self._Ky = kernel

		self._gpModelX = None
		self._gpModelY = None


	def addMeasurement(self, measurement):
		super().addMeasurement(measurement)

	def addMeasurements(self, measurements):
		super().addMeasurements(measurements)

	def clearMeasurements(self):
		super().clearMeasurements()

	def approximate(self, fieldExtents=None):
		if (len(self._measurements) < 1):
			print("No Measurements Available")
			return None

		X = []
		vX = []
		vY = []

		print("Processing ", len(self._measurements), " Measurements")

		for m in self._measurements:
			X.append(m.point)
			vel = m.vector
			vX.append((vel[0]))
			vY.append((vel[1]))


		x = np.asarray(X)
		y1 = np.asarray(vX)
		y2 = np.asarray(vY)
		y1 = np.reshape(y1, (len(vX),1))
		y2 = np.reshape(y2, (len(vY),1))

		meanFuncX = GPflow.mean_functions.Constant()
		#GPy.mappings.Constant(2, 1, np.mean(y1))
		meanFuncY = GPflow.mean_functions.Constant()
		#GPy.mappings.Constant(2, 1, np.mean(y2))

		self._gpModelX = GPflow.gpr.GPR(x, y1, self._Kx, meanFuncX)
		#GPy.models.GPRegression(x, y1, self._Kx, normalizer=True, mean_function=meanFuncX)
		self._gpModelY = GPflow.gpr.GPR(x, y2, self._Ky, meanFuncY)
		#GPy.models.GPRegression(x, y2, self._Ky, normalizer=True, mean_function=meanFuncY)

		#print(self._gpModelX)
		#print(self._gpModelY)
		self._gpModelX.optimize()
		self._gpModelY.optimize()
		#self._gpModelX.randomize()
		#self._gpModelY.randomize()
		#self._gpModelX.optimize_restarts(messages=False, optimizer='tnc', robust=True, num_restarts=1, max_iters=300)
		#self._gpModelY.optimize_restarts(messages=False, optimizer='tnc', robust=True, num_restarts=1, max_iters=300)
		#self._gpModelX.optimize_restarts(messages=False, optimizer='lbfgsb', robust=True, num_restarts=1, max_iters=300)
		#self._gpModelY.optimize_restarts(messages=False, optimizer='lbfgsb', robust=True, num_restarts=1, max_iters=300)
		#self._gpModelX.optimize_restarts(messages=False, optimizer='scg', robust=True, num_restarts=2, max_iters=300)
		#self._gpModelY.optimize_restarts(messages=False, optimizer='scg', robust=True, num_restarts=2, max_iters=300)

		#print(self._gpModelX)
		#print(self._gpModelY)

		#self._gpModel.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})

		vfRep = vf.gpflow_representation.GPFlowVectorFieldRepresentation(self._gpModelX, self._gpModelY, fieldExtents)

		return vf.fields.VectorField(vfRep)