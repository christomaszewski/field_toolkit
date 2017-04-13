import numpy as np
from scipy.integrate import ode

from .. import core

class ParticleDriftSimulator(object):
	""" Basic simulator that uses an ODE solver to propagate a particle through a vector
		field. Assumes massless particle

	"""

	def __init__(self, vectorField=None, noise=0.0001):
		self._flowField = vectorField

		# todo make this gaussian noise
		self._observationNoise = noise


	def simulate(self, seedParticles, time, timestep=0):
		if (self._flowField is None):
			return []

		particleTracks = []

		f = lambda t, y, arg1: list(self._flowField.sampleAtPoint(tuple(y)))
		r = ode(f).set_integrator('dop853')

		for particle in seedParticles:
			y0 = np.asarray(particle)
			t0 = 0

			r.set_initial_value(y0, t0).set_f_params(1.0)

			particleTracks.append(r.integrate(time, timestep))

		return particleTracks

	def changeField(self, newField):
		self._flowField = newField