from unittest import TestCase

import field_toolkit.core.fields as field_lib
from field_toolkit.core.extents import FieldExtents

class VectorFieldTest(TestCase):

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
		channelWidth = 100
		self._vMax = 3.0

		# Domain Description
		xOrigin = 0 #meters
		yOrigin = 0 #meters
		xDist = channelWidth #meters 
		yDist = 50 #meters

		# Build Field Extents from Domain Description
		self._domainExtents = FieldExtents.from_bounds_list([xOrigin, xDist, yOrigin, yDist])

		self._vectorField = field_lib.DevelopedPipeFlowField(channelWidth, self._vMax, self._domainExtents)

	def test_vector_field_construction(self):
		self.assertIsInstance(self._vectorField, field_lib.VectorField)

	def test_vector_field_sampling(self):
		xMin, xMax = self._domainExtents.xRange
		yMin, yMax = self._domainExtents.yRange

		xCenter = (xMin + xMax)/2
		yCenter = (yMin + yMax)/2

		# Sampling in middle of channel flow should be vMax
		vX, vY = self._vectorField.sampleAtPoint((xCenter, yCenter))
		self.assertAlmostEqual(self._vMax, vY)