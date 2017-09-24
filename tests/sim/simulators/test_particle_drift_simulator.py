from unittest import TestCase
import numpy as np
from scipy.integrate import odeint

from field_toolkit.core.fields import VectorField, CompoundVectorField
from field_toolkit.core.extents import FieldExtents
from field_toolkit.sim.simulators import ParticleDriftSimulator

class ParticleDriftTest(TestCase):

	def test_drift_in_compound_field(self):
		# Scenario parameters
		channelWidth = 100 #meters
		maxVelocity = 2 #m/s
		uniformVector = (0.2, -0.5)
		time = 60
		times = np.linspace(0, time, 50)

		# Domain Description
		xOrigin = 0 #meters
		yOrigin = 0 #meters
		xDist = channelWidth #meters
		yDist = 50 #meters
		
		# Build Field Extents from Domain Description
		self._domainExtents = FieldExtents.from_bounds_list([xOrigin, xDist, yOrigin, yDist])

		# Define fully developed pipe flow field
		pipeFlow = VectorField.from_developed_pipe_flow_model(channelWidth, maxVelocity, self._domainExtents, (xOrigin, yOrigin))

		# Define uniform flow field
		uniformFlow = VectorField.from_uniform_vector(uniformVector, self._domainExtents)

		# Define compound field
		self._vectorField = CompoundVectorField(pipeFlow, uniformFlow)

		# Initialize particle simulator with field and 0 noise
		sim = ParticleDriftSimulator(self._vectorField, 0.0)

		# Initialize particle to track
		particle = np.array((xDist/10, yDist/6))

		# Simulate particle drift for time secs
		simTrack = sim.simulateODEInt(particle, times)
		#simTrack = sim.simulate(particle, time)

		# Prepare function for numerical integration
		df = lambda f, t: list((uniformVector[0], uniformVector +\
			((4 * (f[0] - xOrigin) / channelWidth - 4 * (f[0] - xOrigin)**2 / channelWidth**2) * maxVelocity)))
		
		f0 = np.asarray(particle)

		#solution = odeint(df, f0, times)

		dx, dy = uniformVector
		x0, y0 = particle
		v = maxVelocity
		c = channelWidth
		cc = np.square(channelWidth)
		y = lambda t: -4.*v*np.power(t, 3)/(75.*cc) +\
						(4.*c*v-8.*x0*v)*np.square(t)/(10.*cc) +\
						(4.*x0*c*v-4.*x0*x0+cc*dy)*t/cc + y0

		x = lambda t: x0 + dx * t

		maxError = 2.5

		for p, t in zip(simTrack.positions, simTrack.times):
			print(t, p, [x(t), y(t)])
			p2 = np.array((x(t), y(t)))
			error = np.linalg.norm(p-p2)
			self.assertLess(error, maxError)


	def test_drift_in_uniform_field(self):
		# Scenario parameters
		flowVector = (0.8, 0.3) #m/s
		time = 60 #s

		# Domain Description
		xOrigin = 0 #meters
		yOrigin = 0 #meters
		xDist = 100 #meters 
		yDist = 50 #meters

		# Build Field Extents from Domain Description
		self._domainExtents = FieldExtents.from_bounds_list([xOrigin, xDist, yOrigin, yDist])

		# Define uniform flow field on specified extents
		self._vectorField = VectorField.from_uniform_vector(flowVector, self._domainExtents)

		# Initialize particle simulator with field and 0 noise
		sim = ParticleDriftSimulator(self._vectorField, 0.0)

		# Initialize particles to track
		particles = [(xDist/2, yDist/2), (xDist/4, yDist/4)]

		# Simulate particle drift for time secs
		simParticles = [sim.compute(p, time) for p in particles]

		# Compute desired particle locations using known flow vector and time
		predict = lambda p: np.asarray(p) + np.asarray(flowVector) * time
		predictedParticles = [predict(p) for p in particles]

		for simulated, predicted in zip(simParticles, predictedParticles):
			self.assertAlmostEqual(simulated[0], predicted[0])
			self.assertAlmostEqual(simulated[1], predicted[1])