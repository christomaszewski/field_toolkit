from abc import ABCMeta, abstractmethod

class Extents(metaclass=ABCMeta):

	@abstractmethod
	def contain(self, point):
		""" All subclasses should provide method to check whether 
			extents contain a point
		"""
		raise NotImplementedError()

class FieldRepresentation(metaclass=ABCMeta):

	@abstractmethod
	def __getitem__(self, index):
		"""Method to sample field at given point

		Args:
			index (2-tuple): point at which to sample the field

		Returns:
			value (tuple): value sampled from field at index

		"""
		pass

	@property
	@abstractmethod
	def validExtents(self):
		"""All subclasses must provide the extents over which the field
		is defined

		"""
		raise NotImplementedError()

class Field(metaclass=ABCMeta):

	@property
	@abstractmethod
	def representation(self):
		"""All subclasses must provide a field representation

		"""
		raise NotImplementedError()