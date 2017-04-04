import numpy as np
import time

class Measurement(object):
	""" Represents a single measurement of a vector field at a point

	"""

	def __init__(self, point, vector, score=0.0):
		self._point = point
		self._vector = vector
		self._score = score

	@property
	def point(self):
		return self._point

	@property
	def vector(self):
		return self._vector
	
	@property
	def score(self):
		return self._score

	def __cmp__(self, other):
		# Negative so we can use minheap as maxheap
		return -(cmp(self.score, other.score))

	def __lt__(self, other):
		return self.score < other.score

	def __gt__(self, other):
		return self.score > other.score

	def __radd__(self, other):
		return self.score + other

	def __add__(self, other):
		return self.score + other

	def __str__(self):
		return "[" + str(self.point) + "," + str(self.vector) + "]"

	def __repr__(self):
		return str(self)

class SampleGrid(object):
	""" Convenience class to represent a grid to be sampled over.
		Also useful for plotting with quiver

		note: Dimensions all in meters or number of cells

		todo: make this grid offset properly
	"""

	def __init__(self, xDistance, yDistance, xCellCount, yCellCount=None):
		self._xDist = xDistance 							#meters
		self._yDist = yDistance							#meters
		self._xCellCount = xCellCount						#cells

		if (yCellCount is None):
			self._yCellCount = xCellCount					#cells
		else:
			self._yCellCount = yCellCount					#cells

		self._totalCells = self._xCellCount * self._yCellCount

		self._xCellWidth = xDistance / xCellCount			#meters
		self._xCellHalfWidth = self._xCellWidth / 2.0		#meters
		self._yCellWidth = yDistance / yCellCount			#meters
		self._yCellHalfWidth = self._yCellWidth / 2.0		#meters

		self._xGrid, self._yGrid = self.generateGrid()

	@classmethod
	def from_extents(cls, fieldExtents, cellSize=(5,5)):
		# Currently wont work for extents offset from the origin!!!
		# By default the grid density is computed to make each cell 5x5 meters

		xMin, xMax = fieldExtents.xRange
		yMin, yMax = fieldExtents.yRange

		xDist = xMax - xMin
		yDist = yMax - yMin

		xCells = int(xDist / cellSize[0])
		yCells = int(yDist / cellSize[1])

		return cls(xDist, yDist, xCells, yCells)


	def generateGrid(self):
		""" Generates mgrid for used with quiver

		"""
		return np.mgrid[self._xCellHalfWidth:(self._xDist - self._xCellHalfWidth):(self._xCellCount * 1j), 
						self._yCellHalfWidth:(self._yDist - self._yCellHalfWidth):(self._yCellCount * 1j)]


	@property
	def mgrid(self):
		return (self._xGrid, self._yGrid)

	@property
	def arange(self):
		return (np.arange(self._xCellHalfWidth, self._xDist, self._xCellWidth), 
				np.arange(self._yCellHalfWidth, self._yDist, self._yCellWidth))

	@property
	def cellCenters(self):
		xRange, yRange = self.arange
		centers = [(x, y) for x in xRange for y in yRange]
		return centers

	@property
	def size(self):
		return self._totalCells