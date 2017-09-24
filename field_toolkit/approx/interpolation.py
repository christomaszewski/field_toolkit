from .base import FieldApproximator

class InterpolationBasedApproximator(FieldApproximator):
	
	def __init__(self):
		self._measurements = []

	def addMeasurement(self, measurement):
		super().addMeasurement(measurement)

	def addMeasurements(self, measurements):
		super().addMeasurements(measurements)

	def clearMeasurements(self):
		super().clearMeasurements()

	def approximate(self):
		return None