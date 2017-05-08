import numpy as np

from .base import Extents

class FieldExtents(Extents):
	"""Class representing the valid extents of a 2D field

	Provides methods for checking whether a given point is within the defined
	region of a field for use in computing the resultant field value at the
	point in potentially mixed and overlapping fields.

	Attributes:
		_xMin (double): Minimum x value defined
		_xMax (double): Maximum x value defined
		_yMin (double): Minimum y value defined
		_yMax (double): Maximum y value defined

	"""

	def __init__(self, xRange=(0.0, 0.0), yRange=None):
		"""Extracts and stores valid field extents

		Args:
			xRange (2-Tuple): (xMin, xMax)
			yRange (2-Tuple): (yMin, yMax) defaults to xRange when not provided
		"""

		if (yRange is None):
			yRange = xRange

		self._xMin, self._xMax = xRange
		self._yMin, self._yMax = yRange

	@classmethod
	def from_bounds_list(cls, bounds):
		return cls(tuple(bounds[:2]), tuple(bounds[2:]))

	@classmethod
	def from_contained_points(cls, points):
		# Creates extents that surround list of provided points
		p = np.asarray(points)
		xRange = (min(p[:,0]), max(p[:,0]))
		yRange = (min(p[:,1]), max(p[:,1]))

		return cls(xRange, yRange)

	def contain(self, point):
		x, y = point
		if (self._xMin <= x <= self._xMax) and (self._yMin <= y <= self._yMax):
			return True

		return False

	def xSplit(self, *args):
		subExtents = []

		# Need to add sorting of args in case unsorted axes are provided

		# Initialize bounds of remaining extents to be paritioned
		xLower = self._xMin
		xUpper = self._xMax

		for axis in args:
			# if axis is within remaining extents to be partitioned
			if (xLower < axis < xUpper):
				subExtents.append(FieldExtents((xLower, axis), self.yRange))
				# Move xLower bound up to axis
				xLower = axis

				
		subExtents.append(FieldExtents((xLower, xUpper), self.yRange))

		return subExtents

	def ySplit(self, *args):
		subExtents = []

		# Initialize bounds of remaining extents to be paritioned
		yLower = self._yMin
		yUpper = self._yMax

		for axis in args:
			# if axis is within remaining extents to be partitioned
			if (yLower < axis < yUpper):
				subExtents.append(FieldExtents(self.xRange, (yLower, axis)))
				# Move yLower bound up to axis
				yLower = axis

				
		subExtents.append(FieldExtents(self.xRange, (yLower, yUpper)))

		return subExtents

	@property
	def xRange(self):
		return (self._xMin, self._xMax)

	@property
	def yRange(self):
		return (self._yMin, self._yMax)

	@property
	def xDist(self):
		return self._xMax - self._xMin

	@property
	def yDist(self):
		return self._yMax - self._yMin

	@property
	def size(self):
		return(self.xDist, self.yDist)

	@property
	def bounds(self):
		return [self._xMin, self._xMax, self._yMin, self._yMax]

class PiecewiseExtents(FieldExtents):
	"""Class representing the valid extents of a 2D field defined as set of
	component subextents. Does not consider the space between component extents
	as valid.

	"""

	def __init__(self, extents=None):
		if (extents is not None):
			self._subExtents = [extents]
		else:
			self._subExtents = []

	def addExtents(self, extents):
		self._subExtents.append(extents)

	def contain(self, point):
		# same as this?
		#return any([e.contain(point) for e in self._subExtents])

		for extents in self._subExtents:
			if (extents.contain(point)):
				return True

		return False


	@property
	def xRange(self):
		return None

	@property
	def yRange(self):
		return None

	@property
	def xDist(self):
		return None

	@property
	def yDist(self):
		return None

	@property
	def size(self):
		return None

class EncompassingExtents(FieldExtents):
	"""Class representing the valid extents of a 2D field which grow to
	encompass component extents as they are added. This results in the 
	space between component extents being defined as valid.

	Note:
		Does not currently handle infinite extents!!!

	"""

	def __init__(self, extents):
		self._xMin, self._xMax = extents.xRange
		self._yMin, self._yMax = extents.yRange

		self._subExtents = [extents]

	def addExtents(self, extents):
		self._subExtents.append(extents)

		xMin, xMax = extents.xRange
		yMin, yMax = extents.yRange

		self._xMin = min(self._xMin, xMin)
		self._yMin = min(self._yMin, yMin)
		self._xMax = max(self._xMax, xMax)
		self._yMax = max(self._yMax, yMax)


class InfiniteExtents(FieldExtents):
	"""Class representing infinite field extents (i.e. all points valid)

	"""

	def __init__(self):
		"""No arguments are necessary, extents are infinite

		"""
		pass

	def contain(self, point):
		return True

	@property
	def xRange(self):
		return None

	@property
	def yRange(self):
		return None

	@property
	def xDist(self):
		return None

	@property
	def yDist(self):
		return None

	@property
	def size(self):
		return None

class NullExtents(FieldExtents):
	"""Class representing null field extents (i.e. no points valid)

	"""

	def __init__(self):
		"""No arguments are necessary, extents are null
		xMin set to greater than xMax 
		yMin set to greater than yMax

		"""
		self._xMin = 1.0
		self._xMax = -1.0
		self._yMin = 1.0
		self._yMax = -1.0

	def contain(self, point):
		# For efficiency, should return False with super class contain method
		return False