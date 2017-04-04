import numpy as np
import matplotlib.pyplot as plt

from .. import core
from ..core import primitives

class GridSampleEvaluator(object):

	def __init__(self, sampleGrid, sourceField=None, approxField=None):
		self._grid = sampleGrid
		self._source = sourceField
		self._approx = approxField

		self._error = None
		self._xSource = self._ySource = None
		self._xApprox = self._yApprox = None

		if (self._source is not None):
			self._updateGroundTruth()

		if (self._approx is not None):
			self._updateApprox()

		self._computeErrors()

	def setGrid(self, sampleGrid):
		self._grid = sampleGrid
		self._error = None
		self._updateGroundTruth()
		self._updateApprox()
		self._computeErrors()

	def setGroundTruth(self, sourceField):
		self._source = sourceField
		self._error = None
		self._updateGroundTruth()
		self._computeErrors()

	def setApprox(self, approxField):
		self._approx = approxField
		self._error = None
		self._updateApprox()
		self._computeErrors()

	def _updateGroundTruth(self):
		self._xSource, self._ySource = self._source.sampleGrid(self._grid)
		xSquared = self._xSource ** 2
		ySquared = self._ySource ** 2
		self._summedMag = np.sum(np.sqrt(xSquared + ySquared))

	def _updateApprox(self):
		self._xApprox, self._yApprox = self._approx.sampleGrid(self._grid)

	def _computeErrors(self):
		if (self._source is None or self._approx is None):
			self._error = None
			return False

		if (self._xSource is None or self._ySource is None):
			self._updateGroundTruth()

		if (self._xApprox is None or self._yApprox is None):
			self._updateApprox()

		self._xDiff = self._xApprox - self._xSource
		self._yDiff = self._yApprox - self._ySource
		self._squareDiffX = self._xDiff ** 2
		self._squareDiffY = self._yDiff ** 2
		
		self._sumSquaredDiffX = np.sum(self._squareDiffX)
		self._sumSquaredDiffY = np.sum(self._squareDiffY)

		summed = np.sum(np.sqrt(self._squareDiffX + self._squareDiffY))

		self._error = summed/self._summedMag

	@property
	def error(self):
		if (self._error is None):
			self._computeErrors()

		return self._error



class GridSampleComparison(object):
	""" Approximation quality evaluation via sampling both source and approximate fields
		across a grid and computing the differences between components of the vectors at
		each sample point

	"""

	def __init__(self, sampleGrid, sourceField=None, approxField=None):
		self._grid = sampleGrid
		self._source = sourceField
		self._approx = approxField

		self._compute()

	def invalidate(self):
		""" Invalidate/clear all computed values
		
		"""

		self._xDiff = None
		self._yDiff = None
		self._sumSquaredDiffX = None
		self._sumSquaredDiffY = None

	def changeFields(self, sourceField=None, approxField=None):
		if (sourceField is not None):
			self._source = sourceField
			self.invalidate()

		if (approxField is not None):
			self._approx = approxField
			self.invalidate()

	def changeGrid(self, sampleGrid):
		self._grid = sampleGrid
		self.invalidate()

	def plotErrors(self):
		if (not self._compute()):
			return
			
		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title('Error Vectors')
		xGrid, yGrid = self._grid.mgrid
		magnitudes = np.sqrt(self._xDiff**2 + self._yDiff**2)
		clim = [magnitudes.min(), magnitudes.max()]
		self._q = self._ax.quiver(xGrid, yGrid, self._xDiff, self._yDiff, magnitudes, 
						clim=clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)

		self._ax.axis(self._source.plotExtents)
		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')
		c = self._fig.colorbar(self._q, ax=self._ax)
		c.set_label('m/s')

		plt.show()
		plt.pause(0.0001)

	def save(self, fileName):
		self._fig.savefig(fileName, bbox_inches='tight', dpi=100)

	def _compute(self):
		# Check if computation is possible
		if (self._source is None or self._approx is None or self._grid is None):
			self.invalidate()
			return False

		xSource, ySource = self._source.sampleGrid(self._grid)
		xSquared = xSource * xSource
		ySquared = ySource * ySource
		self._summedMag = np.sum(np.sqrt(xSquared + ySquared))

		xApprox, yApprox = self._approx.sampleGrid(self._grid)

		self._xDiff = xApprox - xSource
		self._yDiff = yApprox - ySource
		self._squareDiffX = self._xDiff * self._xDiff
		self._squareDiffY = self._yDiff * self._yDiff
		
		self._sumSquaredDiffX = np.sum(self._squareDiffX)
		self._sumSquaredDiffY = np.sum(self._squareDiffY)

		return True

	@property
	def approxError(self):
		if (self._xDiff is None or self._yDiff is None):
			self._compute()

		summed = np.sum(np.sqrt(self._squareDiffX + self._squareDiffY))

		return summed/self._summedMag

	@property
	def error(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()

		return (self._sumSquaredDiffX, self._sumSquaredDiffY)

	@property
	def maxError(self):
		if (self._xDiff is None or self._yDiff is None):
			self._compute()

		return (np.max(np.abs(self._xDiff)), np.max(np.abs(self._yDiff)))

	@property
	def minError(self):
		if (self._xDiff is None or self._yDiff is None):
			self._compute()

		return (np.min(np.abs(self._xDiff)), np.min(np.abs(self._yDiff)))

	@property
	def meanError(self):
		if (self._xDiff is None or self._yDiff is None):
			self._compute()

		return (np.mean(self._xDiff), np.mean(self._yDiff))

	@property
	def errorStd(self):
		if (self._xDiff is None or self._yDiff is None):
			self._compute()

		return (np.std(self._xDiff), np.std(self._yDiff))		

	@property
	def normalError(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()
		
		return (self._sumSquaredDiffX / self._grid.size, 
				self._sumSquaredDiffY / self._grid.size)

	@property
	def errorVectors(self):
		if (self._xDiff is None or self._yDiff is None):
			self._compute()

		return (self._xDiff, self._yDiff)


"""
class StreamLineComparison(object):

	def __init__(self, seedParticles=None, sourceField=None, approxField=None, simTime=1, simRes=0.1):
		self._particles = seedParticles
		self._simTime = simTime
		self._simResolution = simRes

		self._source = sourceField
		self._sourceSim = vf_sim.simulators.ParticleSimulator(self._source, noise=0)

		self._approx = approxField
		self._approxSim = vf_sim.simulators.ParticleSimulator(self._approx, noise=0)
		
		self.invalidate()

		self._trackLength = 0

	def invalidate(self):
		#Invalidate/clear all computed values
		

		self._xDiff = None
		self._yDiff = None
		self._sumSquaredDiffX = None
		self._sumSquaredDiffY = None

	def changeFields(self, sourceField=None, approxField=None):
		if (sourceField is not None):
			self._source = sourceField
			self._sourceSim.changeField(self._source)
			self.invalidate()

		if (approxField is not None):
			self._approx = approxField
			self._approxSim.changeField(self._approx)
			self.invalidate()

	def changeParticles(self, seedParticles):
		self._particles = seedParticles
		self.invalidate()

	def plotErrors(self, grid, field=None):
		tracks = None
		title = ''
		color = 'red'
		if (field is None):
			field = self._approx
			tracks = self._approxTracks[20:505:15]
			title = 'Approximation '
		else:
			field = self._source
			tracks = self._sourceTracks[20:505:15]
			title = 'Source '
			color = 'blue'

		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title(title + 'Streamlines')
		xSamples, ySamples = field.sampleGrid(grid)
		xGrid, yGrid = grid.mgrid
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)
		clim = [magnitudes.min(), magnitudes.max()]
		self._q = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes, 
						clim=clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)

		self._ax.axis(self._source.plotExtents)
		self._ax.hold(True)
		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')
		c = self._fig.colorbar(self._q, ax=self._ax)
		c.set_label('m/s')

		index = 0

		for t in tracks:
			ptSeq = np.asarray(t.getPointSequence())
			if (index == 20):
				self._ax.scatter(ptSeq[:,0], ptSeq[:,1], c='green', edgecolor='green', marker='o')
			else:
				self._ax.scatter(ptSeq[:,0], ptSeq[:,1], c=color, edgecolor=color, marker='o')

			index += 1

		plt.show()
		plt.pause(0.0001)

	def plotStreamlineComparison(self):
		approxTracks = self._approxTracks[20:505:15]
		sourceTracks = self._sourceTracks[20:505:15]

		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title('Streamline Error')
		self._ax.hold(True)
		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')
		aTrack = approxTracks[20]
		aPts = np.asarray(aTrack.getPointSequence())
		sTrack = sourceTracks[20]
		sPts = np.asarray(sTrack.getPointSequence())

		s1 = self._ax.scatter(aPts[:,0], aPts[:,1], c='red', edgecolor='red', marker='o', label='Approximation Streamline')
		s2 = self._ax.scatter(sPts[:,0], sPts[:,1], c='blue', edgecolor='blue', marker='o', label='Source Streamline')
		plt.legend(handles=[s1, s2])
		for a,s in zip(aTrack.getPointSequence(), sTrack.getPointSequence()):
			self._ax.plot([a[0], s[0]], [a[1], s[1]], c='black')


	def save(self, fileName):
		self._fig.savefig(fileName, bbox_inches='tight', dpi=100)


	def _compute(self):
		# Check if computation is possible
		if (self._source is None or self._approx is None or self._particles is None):
			self.invalidate()
			return

		self._sourceTracks = self._sourceSim.simulate(self._particles, self._simTime, self._simResolution)
		self._approxTracks = self._approxSim.simulate(self._particles, self._simTime, self._simResolution)
		self._trackLength = self._sourceTracks[0].size()

		differences = [a - s for a, s in zip(self._approxTracks, self._sourceTracks)]
		
		self._xDiff = 0.0
		self._yDiff = 0.0
		self._squaredDiffX = 0.0
		self._squaredDiffY = 0.0

		for d in differences:
			# For each d we want to create a pair of error terms
			diff = np.asarray(d)
			squaredDiff = diff * diff

			self._xDiff += np.sum(diff[:, 0])
			self._yDiff += np.sum(diff[:, 1])
			self._squaredDiffX += np.sum(squaredDiff[:, 0])
			self._squaredDiffY += np.sum(squaredDiff[:, 1])
			
		
		self._sumSquaredDiffX = self._squaredDiffX
		self._sumSquaredDiffY = self._squaredDiffY


	@property
	def error(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()

		return (self._sumSquaredDiffX / len(self._particles),
				self._sumSquaredDiffY / len(self._particles))

	@property
	def normalError(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()
		
		return (self._sumSquaredDiffX / (len(self._particles) * self._trackLength), 
				self._sumSquaredDiffY / (len(self._particles) * self._trackLength))

"""