import GPy
import numpy as np

from .. import core
from .base import FieldApproximator

class GPApproximator(FieldApproximator):

	def __init__(self, kernel=None):
		self._measurements = []

		if (kernel is None):
			# Default kernel
			
			self._Kx = GPy.kern.Matern52(input_dim=1, active_dims=[0], lengthscale=20, ARD=True) *\
						GPy.kern.Matern52(input_dim=1, active_dims=[1], lengthscale=20, ARD=True) +\
						GPy.kern.RatQuad(input_dim=2, lengthscale=100, ARD=True) +\
						GPy.kern.White(input_dim=2)
			self._Ky = GPy.kern.Matern52(input_dim=1, active_dims=[0], lengthscale=20, ARD=True) *\
						GPy.kern.Matern52(input_dim=1, active_dims=[1], lengthscale=500, ARD=True) +\
						GPy.kern.RatQuad(input_dim=2, lengthscale=100, ARD=True) +\
						GPy.kern.White(input_dim=2)
			"""
			self._Kx = GPy.kern.Matern52(input_dim=2, lengthscale=20, ARD=True) +\
						GPy.kern.RatQuad(input_dim=2, lengthscale=80, ARD=True) +\
						GPy.kern.White(input_dim=2)
			self._Ky = GPy.kern.Matern52(input_dim=2, lengthscale=20, ARD=True) +\
						GPy.kern.RatQuad(input_dim=2, lengthscale=(60, 300), ARD=True) +\
						GPy.kern.White(input_dim=2)
			"""
		else:
			self._Kx = kernel[0]
			self._Ky = kernel[1]

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

		if (fieldExtents is None):
			print("No extents specified, building extents from measurements")
			# Generate extents surrounding sample points
			points = [m.point for m in self._measurements]
			fieldExtents = core.extents.FieldExtents.from_contained_points(points)


		X = []
		vX = []
		vY = []

		print("Processing ", len(self._measurements), " Measurements")

		for m in self._measurements:
			X.append(m.point)
			vel = m.value
			vX.append((vel[0]))
			vY.append((vel[1]))


		x = np.asarray(X)
		y1 = np.asarray(vX)
		y2 = np.asarray(vY)
		y1 = np.reshape(y1, (len(vX),1))
		y2 = np.reshape(y2, (len(vY),1))

		meanFuncX = GPy.mappings.Constant(2, 1, 0.0)
		#meanFuncY = GPy.mappings.Constant(2, 1, np.mean(y2))
		meanFuncY = GPy.mappings.Constant(2, 1, np.median(y2))
		print(np.median(y2), np.mean(y2))

		#self._gpModelX = GPy.models.GPRegression(x, y1, self._Kx, normalizer=False, mean_function=meanFuncX)
		#self._gpModelY = GPy.models.GPRegression(x, y2, self._Ky, normalizer=False, mean_function=meanFuncY)

		self._gpModelX = GPy.models.GPRegression(x, y1, self._Kx, normalizer=True, mean_function=meanFuncX)
		self._gpModelY = GPy.models.GPRegression(x, y2, self._Ky, normalizer=True, mean_function=meanFuncY)

		print(self._gpModelX['.*lengthscale'])
		print(self._gpModelY['.*lengthscale'])

		self._gpModelX['.*Mat52.lengthscale'].constrain_bounded(1., 10000.)
		self._gpModelX['.*Mat52_1.lengthscale'].constrain_bounded(1., 10000.)
		self._gpModelX['.*RatQuad.lengthscale'].constrain_bounded(1., 100.)

		self._gpModelY['.*Mat52.lengthscale'].constrain_bounded(1., 10000.)
		self._gpModelY['.*Mat52_1.lengthscale'].constrain_bounded(200., 10000.)
		self._gpModelY['.*RatQuad.lengthscale'].constrain_bounded(1., 100.)

		print(self._gpModelX['.*lengthscale'])
		print(self._gpModelY['.*lengthscale'])

		#print(self._gpModelX)
		#print(self._gpModelY)
		self._gpModelX.randomize()
		self._gpModelY.randomize()
		#self._gpModelX.optimize_restarts(messages=False, optimizer='tnc', robust=True, num_restarts=2, max_iters=500)
		#self._gpModelY.optimize_restarts(messages=False, optimizer='tnc', robust=True, num_restarts=2, max_iters=500)
		self._gpModelX.optimize_restarts(messages=True, optimizer='lbfgsb', robust=True, num_restarts=5, max_iters=2000)
		self._gpModelY.optimize_restarts(messages=True, optimizer='lbfgsb', robust=True, num_restarts=5, max_iters=2000)
		#self._gpModelX.optimize_restarts(messages=True, optimizer='scg', robust=True, num_restarts=2, max_iters=500)
		#self._gpModelY.optimize_restarts(messages=True, optimizer='scg', robust=True, num_restarts=2, max_iters=500)

		print(self._gpModelX)
		print(self._gpModelY)


		print(self._gpModelX['.*lengthscale'])
		print(self._gpModelY['.*lengthscale'])
		#self._gpModel.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})

		vfRep = core.gp_representation.GPVectorFieldRepresentation(self._gpModelX, self._gpModelY, fieldExtents)

		return core.fields.VectorField(vfRep)

class IntegralGPApproximator(FieldApproximator):
	""" Does not seem to work reliably at the moment
	"""
	def __init__(self, kernel=None):
		self._measurements = []

		if (kernel is None):
			# Default kernel
			self._Kx = GPy.kern.Integral(input_dim=2, ARD=True) +\
						GPy.kern.White(input_dim=2)

			self._Ky = GPy.kern.Integral(input_dim=2, ARD=True) +\
						GPy.kern.White(input_dim=2)
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
			vel = m.point #Different for integral kernel
			vX.append((vel[0]))
			vY.append((vel[1]))


		x = np.asarray(X)
		y1 = np.asarray(vX)
		y2 = np.asarray(vY)
		y1 = np.reshape(y1, (len(vX),1))
		y2 = np.reshape(y2, (len(vY),1))


		self._gpModelX = GPy.models.GPRegression(x, y1, self._Kx, normalizer=False)
		self._gpModelY = GPy.models.GPRegression(x, y2, self._Ky, normalizer=False)

		#print(self._gpModelX)
		#print(self._gpModelY)
		self._gpModelX.randomize()
		self._gpModelY.randomize()

		self._gpModelX.optimize_restarts(messages=False, optimizer='lbfgsb', robust=True, num_restarts=2, max_iters=10000)
		self._gpModelY.optimize_restarts(messages=False, optimizer='lbfgsb', robust=True, num_restarts=2, max_iters=10000)
		#self._gpModelX.optimize_SGD()
		#self._gpModelY.optimize_SGD()
		#print(self._gpModelX)
		#print(self._gpModelY)

		#self._gpModel.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})

		vfRep = core.gp_representation.GPVectorFieldRepresentation(self._gpModelX, self._gpModelY, fieldExtents)

		return core.fields.VectorField(vfRep)

class SparseGPApproximator(FieldApproximator):

	def __init__(self, kernel=None):
		self._measurements = []

		if (kernel is None):
			# Default kernel
			self._Kx = GPy.kern.RatQuad(input_dim=2, ARD=True) +\
						GPy.kern.White(input_dim=2)
			self._Ky = GPy.kern.RatQuad(input_dim=2, ARD=True) +\
						GPy.kern.Bias(input_dim=2) * GPy.kern.Matern32(input_dim=2, ARD=True) +\
						GPy.kern.White(input_dim=2)
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
			vel = m.value
			vX.append((vel[0]))
			vY.append((vel[1]))


		x = np.asarray(X)
		y1 = np.asarray(vX)
		y2 = np.asarray(vY)
		y1 = np.reshape(y1, (len(vX),1))
		y2 = np.reshape(y2, (len(vY),1))


		self._gpModelX = GPy.models.SparseGPRegression(x, y1, self._Kx, normalizer=True, num_inducing=100)
		self._gpModelY = GPy.models.SparseGPRegression(x, y2, self._Ky, normalizer=True, num_inducing=100)
		#print(self._gpModelX['inducing_inputs'])
		#print(self._gpModelX)
		#print(self._gpModelY)
		self._gpModelX.randomize()
		self._gpModelY.randomize()

		self._gpModelX.optimize_restarts(messages=False, optimizer='lbfgsb', robust=True, num_restarts=2, max_iters=300)
		self._gpModelY.optimize_restarts(messages=False, optimizer='lbfgsb', robust=True, num_restarts=2, max_iters=300)

		self._gpModelX.optimize_restarts(messages=False, optimizer='scg', robust=True, num_restarts=2, max_iters=300)
		self._gpModelY.optimize_restarts(messages=False, optimizer='scg', robust=True, num_restarts=2, max_iters=300)

		#print(self._gpModelX)
		#print(self._gpModelY)

		#self._gpModel.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})

		vfRep = core.gp_representation.GPVectorFieldRepresentation(self._gpModelX, self._gpModelY, fieldExtents)

		return core.fields.VectorField(vfRep)


class CoregionalizedGPApproximator(FieldApproximator):

	def __init__(self):
		self._measurements = []


		# Bias Kernel
		self._biasK = GPy.kern.Bias(input_dim=2)
		
		# Linear Kernel
		self._linearK = GPy.kern.Linear(input_dim=2, ARD=True)
		
		# Matern 3/2 Kernel
		self._maternK = GPy.kern.Matern32(input_dim=2, ARD=True, lengthscale=50)
		
		self._ratQuadK = GPy.kern.RatQuad(input_dim=2, ARD=True)

		self._whiteK = GPy.kern.White(input_dim=2)

		kList = [self._biasK, self._ratQuadK, self._whiteK]

		# Build Coregionalized
		self._coregionalizedK = GPy.util.multioutput.LCM(input_dim=2, num_outputs=2, kernels_list=kList)

		self._gpModel = None

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
			vel = m.value
			vX.append((vel[0]))
			vY.append((vel[1]))

		x = np.asarray(X)
		y1 = np.asarray(vX)
		y2 = np.asarray(vY)
		y1 = np.reshape(y1, (len(vX),1))
		y2 = np.reshape(y2, (len(vY),1))

		# Coregionalization stuff
		self._gpModel = GPy.models.GPCoregionalizedRegression([x, x], [y1, y2], self._coregionalizedK)
		print(self._gpModel)

		self._gpModel.randomize()

		# Set constraints
		self._gpModel['.*bias.var'].constrain_fixed(1.)
		self._gpModel['.*RatQuad.var'].constrain_fixed(1.)
		self._gpModel['.*white.var'].constrain_fixed(1.)
		self._gpModel['.*ICM2.*W'].constrain_fixed(0.)
		#self._gpModel['.*ICM.*var'].unconstrain()
		#self._gpModel['.*ICM0.*var'].constrain_fixed(1.)
		#self._gpModel['.*ICM0.*W'].constrain_fixed(0)
		#self._gpModel['.*ICM1.*var'].constrain_fixed(1.)
		#self._gpModel['.*ICM1.*W'].constrain_fixed(0)
		#self._gpModel['.*ICM2.*var'].constrain_fixed(1.)


		print(self._gpModel)

		self._gpModel.optimize_restarts(num_restarts=5, robust=True, max_iters=2000, optimizer='lbfgsb')
		#self._gpModel.optimize_restarts(num_restarts=2, robust=True, max_iters=1000, optimizer='scg')

		print(self._gpModel)
		vfRep = core.gp_representation.CoregionalizedGPFieldRepresentation(self._gpModel, fieldExtents)

		return core.fields.VectorField(vfRep)


class ScalarGPApproximator(FieldApproximator):

	def __init__(self, kernel=None):
		self._measurements = []

		if (kernel is None):
			# Default kernel
			self._K = GPy.kern.RatQuad(input_dim=2, ARD=True) +\
						GPy.kern.White(input_dim=2)
		else:
			self._K = kernel

		self._gpModel = None


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

		if (fieldExtents is None):
			print("No extents specified, building extents from measurements")
			# Generate extents surrounding sample points
			points = [m.point for m in self._measurements]
			fieldExtents = core.extents.FieldExtents.from_contained_points(points)


		X = []
		Y = []

		print("Processing ", len(self._measurements), " Measurements")

		for m in self._measurements:
			X.append(m.point)
			Y.append(m.value)

		x = np.asarray(X)
		y = np.asarray(Y)
		y1 = np.reshape(y, (len(y),1))
		print(y1)
		print(np.mean(y1))

		meanFunc = GPy.mappings.Constant(2, 1, np.mean(y1))

		self._gpModel = GPy.models.GPRegression(x, y1, self._K, normalizer=False)#, mean_function=meanFunc)

		#print(self._gpModel)
		self._gpModel.randomize()
		self._gpModel.optimize_restarts(messages=False, optimizer='lbfgsb', robust=True, num_restarts=5, max_iters=1500)
		#self._gpModel.optimize_restarts(messages=False, optimizer='scg', robust=True, num_restarts=2, max_iters=500)

		#print(self._gpModel)

		#self._gpModel.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})

		sfRep = core.gp_representation.GPScalarFieldRepresentation(self._gpModel, fieldExtents)

		return core.fields.VectorField(sfRep)