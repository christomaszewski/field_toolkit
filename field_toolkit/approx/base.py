from abc import ABCMeta, abstractmethod

class VectorFieldApproximator(metaclass=ABCMeta):

	@abstractmethod
	def addMeasurement(self, measurement):
		self._measurements.append(measurement)

	@abstractmethod
	def addMeasurements(self, measurements):
		self._measurements.extend(measurements)

	@abstractmethod
	def clearMeasurements(self):
		self._measurements.clear()

	@abstractmethod
	def approximate(self):
		pass