import numpy as np
from scipy.integrate import ode, odeint

from primitives.track import Track

from .. import core

class ParticleDriftSimulator(object):
	""" Basic simulator that uses an ODE solver to propagate a particle through a vector
		field. Assumes massless particle

	"""

	def __init__(self, vectorField=None, noise=0.0001):
		self._flowField = vectorField

		# todo make this gaussian noise
		self._observationNoise = noise

	def compute(self, particle, time):
		""" Function to compute the position of a particle in a vector field after
			a given amount of time
		"""

		if (self._flowField is None):
			print("Error: No vector field is set")
			return None

		f = lambda t, y: list(self._flowField.sampleAtPoint(tuple(y)))
		r = ode(f).set_integrator('dop853')

		y0 = np.asarray(particle)
		t0 = 0

		r.set_initial_value(y0, t0)
		return r.integrate(time)

	def simulateODEInt(self, particle, times):
		if (self._flowField is None):
			print("Error: No vector field is set")
			return None


		particleObservations = []
		particlePositions = []
		particleTimes = []

		f = lambda y,t: list(self._flowField.sampleAtPoint(tuple(y)))
		
		particleVel = self._flowField.sampleAtPoint(tuple(particle))
		print(f"Initial Particle Velocity: {particleVel}")

		y0 = np.asarray(particle)

		solution = odeint(f, y0, times)

		#print(solution)

		particleTrack = Track(solution, times)
		return particleTrack

	def _motion(self, t, y):
		x, y, dx, dy = y
		ux, uy = self._flowField.sampleAtPoint((x,y))
		
		m = self._mass

		# if mass is very small, treat as massless
		if self._mass < 0.001:
			return [ux, uy, 0., 0.]

		u = np.array((ux, uy))
		v = np.array((dx, dy))


		# stokes drag coefficient - can we learn this?
		bx = 5.5
		by = 0.0275

		dvx, dvy = u-v
		# stokes drag estimate
		ax = bx * dvx / m
		ay = by * dvy / m

		
		# turbulent drag computation
		aa = 0.25 # cross sectional area
		if m < 1.:
			aa = 0.005
		cd = 1.2
		dv = u-v
		#ax, ay = 0.5 * (np.abs(dv) * dv) * cd * aa / m
		
		return [dx, dy, ax, ay]


	def simulateWithMass(self, particle, velocity, mass, endTime, startTime=0.0):
		if (self._flowField is None):
			print("Error: No vector field is set")
			return None

		self._mass = mass
		particleObservations = []
		particlePositions = []
		particleTimes = []

		# y = [x, y, x', y']

		f = lambda t, y: self._motion(t,y)
		r = ode(f).set_integrator('dop853')
		
		x, y = particle
		vx, vy = velocity

		if self._mass < 0.001:
			vx, vy = self._flowField.sampleAtPoint((x,y))
		
		y0 = np.asarray([x,y,vx,vy])
		print("y0:", y0)
		particleVel = self._flowField.sampleAtPoint(tuple(particle))
		print(f"Field Velocity at initial particle location: {particleVel}")

		t0 = startTime
		print(self._motion(t0, y0))
		#particleTimes.append(t0)
		#particlePositions.append(y0)

		savePosition = lambda t, y: particleObservations.append((t,[*y]))
		r.set_solout(savePosition)
		r.set_initial_value(y0, t0)
		r.integrate(endTime)
		

		for t, p in particleObservations:
			particleTimes.append(t)
			particlePositions.append(np.asarray(p[:2]))

		particleTrack = Track(particlePositions, particleTimes)
		return particleTrack


	def simulate(self, particle, endTime, startTime=0.0, timestep=None):
		if (self._flowField is None):
			print("Error: No vector field is set")
			return None


		particleObservations = []
		particlePositions = []
		particleTimes = []

		f = lambda t, y: list(self._flowField.sampleAtPoint(tuple(y)))
		r = ode(f).set_integrator('dop853')
		
		particleVel = self._flowField.sampleAtPoint(tuple(particle))
		#print(f"Initial Particle Velocity: {particleVel}")

		y0 = np.asarray(particle)
		t0 = startTime

		#particleTimes.append(t0)
		#particlePositions.append(y0)

		if (timestep is None):
			savePosition = lambda t, y: particleObservations.append((t,[*y]))
			r.set_solout(savePosition)
			r.set_initial_value(y0, t0)
			r.integrate(endTime)
		else:
			r.set_initial_value(y0, t0)
			while (r.successful() and r.t < endTime):
				y = r.integrate(r.t+timestep)
				particleObservations.append((r.t, [*y]))

		for t, p in particleObservations:
			particleTimes.append(t)
			particlePositions.append(np.asarray(p))

		particleTrack = Track(particlePositions, particleTimes)
		return particleTrack

	def changeField(self, newField):
		self._flowField = newField