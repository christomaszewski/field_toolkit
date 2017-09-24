import numpy as np

from ..sim.simulators import ParticleDriftSimulator

class DriftAnalysis(object):

	def __init__(self, field=None, simulator=None):
		self._field = field

		if (simulator is None):
			self._sim = ParticleDriftSimulator(field)
		else:
			self._sim = simulator

	def updateField(self, field):
		self._field = field

		self._sim.changeField(field)

	def evaluate(self, track, mass=0.0001):
		if self._field is None:
			print("Error: No field specified")
			return

		nextIndex = min(210, track.size()-1)
		midIndex = nextIndex // 2

		startTime, startPoint = track.getFirstObservation()
		nextTime, nextPoint = track[nextIndex]
		endTime, endPoint = track.getLastObservation()

		# Increment end time to avoid any boundary issues
		endTime += 0.001
		velocity = (nextPoint - startPoint) / (nextTime - startTime)
		"""
		v = np.array((0., 0.))
		p0 = startPoint
		t0 = startTime
		t = midIndex
		for idx in range(1,t):
			t1, p1 = track[idx]
			v += ((p1-p0)/(t1-t0))
			t0 = t1
			p0 = p1

		velocity = v / t
		"""
		simTrack = self._sim.simulateWithMass(startPoint, velocity, mass, endTime, startTime)
		#simTrack = self._sim.simulate(startPoint, endTime, startTime)
		#simTrack = self._sim.simulateODEInt(startPoint, track.times)
		#print(simTrack.times)
		differences = []
		errors = []
		normalizedErrors = []
		for p, t in zip(track.positions, track.times):
			diff = p - simTrack.atTime(t)
			age = t - startTime
			err = np.linalg.norm(diff)
			#print(p, t, diff, age, err)
			normErr = 0.0
			if age > 0.99:
				normErr = err / age
				normalizedErrors.append(normErr)


			differences.append(diff)
			errors.append(err)
			#normalizedErrors.append(normErr)


		duration = track.age()
		normTimeOffset = len(errors) - len(normalizedErrors)

		maxError = np.max(errors)
		normalMaxError = np.max(normalizedErrors)
		maxErrorIndex = np.argmax(errors)
		maxNormalErrorIndex = np.argmax(normalizedErrors) + normTimeOffset
		maxErrorTime = track.times[maxErrorIndex]
		maxNormalErrorTime = track.times[maxNormalErrorIndex]

		orig_max_point = track.atTime(maxErrorTime)
		sim_max_point = simTrack.atTime(maxErrorTime)

		print(orig_max_point, sim_max_point)

		x = [orig_max_point[0], sim_max_point[0]]
		y = [orig_max_point[1], sim_max_point[1]]

		return simTrack, x, y, errors, normalizedErrors

