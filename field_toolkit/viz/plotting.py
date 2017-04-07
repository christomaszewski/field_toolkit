import matplotlib.pyplot as plt
import numpy as np

from ..core import primitives

plt.ion()

class SimpleFieldView(object):
	"""Basic vector field plotting functionality

	"""

	def __init__(self, field=None, grid=None, pause=0.0001, autoRefresh=False):
		self._grid = grid
		self._field = field

		plt.ion()
		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = None
		self._q = None

		self._pauseLength = pause

		self._title = "Untitled"
		self._clim = None
		self._annotation = ""
		self._refresh = autoRefresh

	def draw(self):
		plt.show()
		plt.pause(self._pauseLength)

	def setTitle(self, title):
		self._title = title

	def setClim(self, lim):
		self._clim = lim

	def setAnnotation(self, text):
		self._annotation = text

	def quiver(self):
		if (self._field is None):
			return
		elif (self._grid is None):
			self._grid = primitives.SampleGrid.from_extents(self._field.extents)

		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title(self._title + self._annotation)
		#self._ax.grid(True)

		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = self._field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)

		if (self._clim is None):
			self._clim = [magnitudes.min(), magnitudes.max()]

		self._q = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)


		#self._ax.text(85, 48, self._annotation)

		self._ax.axis(self._field.plotExtents)

		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')

		c = self._fig.colorbar(self._q, ax=self._ax)
		c.set_label('m/s')

		self.draw()

		# Force recomputation of colorbar
		self._clim = None

	def plotTrack(self, track, color, marker='o', label='default'):
		#self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.grid(True)
		self._ax.axis([0, 10, 0, 10])
		t = np.asarray(track.getPointSequence())
		self._ax.scatter(t[:,0], t[:,1], c=color, marker=marker, s=150, label=label)
		self._ax.plot(t[:,0], t[:,1], c=color)
		plt.xticks(np.arange(0, 11, 2.0))
		plt.yticks(np.arange(0, 11, 2.0))
		plt.legend()


		self.draw()

	def clearFig(self):
		self._fig.clf()


	def plotMeasurements(self, measurements, color, label):
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.grid(True)
		self._ax.axis([0, 10, 0, 10])
		t = np.asarray([m.point for m in measurements])
		self._ax.scatter(t[:,0], t[:,1], c=color, marker='o', s=150, label=label)
		plt.xticks(np.arange(0, 11, 2.0))
		plt.yticks(np.arange(0, 11, 2.0))
		plt.legend()

	def plotPoints(self, points, color, marker, label):
		t = np.asarray(points)
		self._ax.scatter(t[:,0], t[:,1], c=color, marker=marker, s=150, label=label)
		plt.legend()

	def changeGrid(self, newGrid):
		self._grid = newGrid

		if (self._refresh):
			self.quiver()

	def changeField(self, newField):
		if (newField is not None):
			self._field = newField

		if (self._refresh):
			self.quiver()


	def save(self, fileName="default.png"):
		self._fig.savefig(fileName, bbox_inches='tight', dpi=100)

	@property
	def clim(self):
		return self._clim


class OverlayFieldView(object):
	"""Plotting a vector field over a background image

	"""

	def __init__(self, field=None, grid=None, img=None, pause=0.0001):
		self._field = field
		self._grid = grid

		# Assume incoming image is in opencv format
		if (img is not None):
			self.updateImage(img)
		else:
			self._img = None

		plt.ion()
		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = None
		self._q = None

		self._pauseLength = pause

		self._title = "Untitled"
		self._clim = None
		self._annotation = ""

	def updateImage(self, img):
		# todo: remove reliance on cv2 here and thereby the entire module
		#self._img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self._img = self._img[:, :, ::-1]
		self._img = np.flipud(self._img)

	def draw(self):
		plt.show()
		plt.pause(self._pauseLength)

	def setTitle(self, title):
		self._title = title

	def setClim(self, lim):
		self._clim = lim

	def setAnnotation(self, text):
		self._annotation = text

	def quiver(self):
		if (self._field is None or self._grid is None):
			return

		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title(self._title + self._annotation)

		if (self._img is not None):
			self._ax.imshow(self._img, origin='lower', aspect='auto')

		self._ax.hold(True)

		#self._ax.grid(True)

		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = self._field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)

		if (self._clim is None):
			self._clim = [magnitudes.min(), magnitudes.max()]

		self._q = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)


		#self._ax.text(85, 48, self._annotation)

		self._ax.axis(self._field.plotExtents)

		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')

		c = self._fig.colorbar(self._q, ax=self._ax)
		c.set_label('m/s')
		self.draw()

		# Force recomputation of colorbar
		self._clim = None

	def plotTrack(self, track, color, marker='o'):
		t = np.asarray(track.getPointSequence())
		self._ax.scatter(t[:,0], t[:,1], c=color, marker=marker)
		self.draw()

	def changeGrid(self, newGrid):
		self._grid = newGrid
		self.quiver()

	def changeField(self, newField):
		self._field = newField
		self.quiver()

	def save(self, fileName="default.png"):
		self._fig.savefig(fileName, bbox_inches='tight', dpi=100)

	@property
	def clim(self):
		return self._clim