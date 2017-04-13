from unittest import TestCase

from field_toolkit.core.fields import UniformVectorField
from field_toolkit.core.extents import FieldExtents

class UniformFieldTest(TestCase):

	@classmethod
	def setUpClass(cls):
		pass

	@classmethod
	def tearDownClass(cls):
		pass

	def setUp(self):
		self._constructVectorField()


	# Internal Fixturing Methods 
	def _constructVectorField(self):
		# Scenario Parameters
		self._vector = (3.0, -1.0)

		# Domain Description
		xOrigin = 0 #meters
		yOrigin = 0 #meters
		xDist = 100 #meters 
		yDist = 50 #meters

		# Build Field Extents from Domain Description
		self._domainExtents = FieldExtents.from_bounds_list([xOrigin, xDist, yOrigin, yDist])

		self._vectorField = UniformVectorField(self._vector, self._domainExtents)

	def test_uniform_field_construction(self):
		self.assertIsInstance(self._vectorField, UniformVectorField)

	def test_uniform_field_sampling(self):
		xMin, xMax = self._domainExtents.xRange
		yMin, yMax = self._domainExtents.yRange

		xCenter = (xMin + xMax)/2
		yCenter = (yMin + yMax)/2

		# Sample in the middle of the valid field extents
		vX, vY = self._vectorField.sampleAtPoint((xCenter, yCenter))

		self.assertAlmostEqual(self._vector[0], vX)
		self.assertAlmostEqual(self._vector[1], vY)