from unittest import TestCase
import numpy as np

from field_toolkit.core.fields import UniformVectorField
from field_toolkit.core.extents import FieldExtents
from field_toolkit.sim.simulators import ParticleDriftSimulator

class ParticleDriftTest(TestCase):

	def test_drift_in_uniform_field(self):
		# Scenario parameters
		flowVector = (1, 2) #m/s
		time = 5 #s

		# Domain Description
		xOrigin = 0 #meters
		yOrigin = 0 #meters
		xDist = 100 #meters 
		yDist = 50 #meters

		# Build Field Extents from Domain Description
		self._domainExtents = FieldExtents.from_bounds_list([xOrigin, xDist, yOrigin, yDist])

		# Define uniform flow field on specified extents
		self._vectorField = UniformVectorField(flowVector, self._domainExtents)

		# Initialize particle simulator with field and 0 noise
		sim = ParticleDriftSimulator(self._vectorField, 0.0)

		# Initialize particles to track
		particles = [(xDist/2, yDist/2), (xDist/4, yDist/4)]

		# Simulate particle drift for time secs
		simParticles = sim.simulate(particles, time)

		# Compute desired particle locations using known flow vector and time
		predict = lambda p: np.asarray(p) + np.asarray(flowVector) * time
		predictedParticles = [predict(p) for p in particles]

		for simulated, predicted in zip(simParticles, predictedParticles):
			self.assertAlmostEqual(simulated[0], predicted[0])
			self.assertAlmostEqual(simulated[1], predicted[1])