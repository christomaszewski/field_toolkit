from . import extents
from .base import FieldRepresentation

class ScalarFieldRepresenation(FieldRepresentation):
	"""Specific class implementation for a 2D Scalar field for which a 
	closed form solution is known

	"""

	def __init__(self, function, fieldExtents, undefinedValue=0.0):
		self._fieldFunc = function
		self._validExtents = fieldExtents
		self._undefinedVal = undefinedValue

	def __getitem__(self, index):
		if (self._validExtents.contain(index)):
			x, y = index
			return self._fieldFunc(x, y)
		else:
			return self._undefinedVal

	def isDefinedAt(self, point):
		return self._validExtents.contain(point)

	@property
	def validExtents(self):
		return self._validExtents


class VectorFieldRepresentation(FieldRepresentation):
	"""Specific class implementation for a 2D vector field for which a
	closed form solution is known

	"""

	def __init__(self, function, fieldExtents, undefinedValue=(0.0, 0.0)):
		"""Stores closed form represenation of vector field and valid extents

		Args:
			function (func): function mapping points in the field to vector values
			extents (FieldExtents): extents over which function is valid/applies
			undefinedValue (2-Tuple): value to return when sampling outside of extents

		"""
		self._fieldFunc = function
		self._validExtents = fieldExtents
		self._undefinedVal = undefinedValue

	def __getitem__(self, index):
		if (self._validExtents.contain(index)):
			# if in valid region, return function value
			x, y = index
			return self._fieldFunc(x, y)
		else:
			# not in valid region, return default value
			return self._undefinedVal

	def isDefinedAt(self, point):
		return self._validExtents.contain(point)

	@property
	def validExtents(self):
		return self._validExtents

	@validExtents.setter
	def validExtents(self, newExtents):
		self._validExtents = newExtents


class CompoundVectorFieldRepresentation(FieldRepresentation):
	"""Class implementation for  a vector field made up of a group of component
	vector fields. Handles all the extents checking and individual value lookups
	for computing the resultant vector field

	"""

	def __init__(self, fieldRep=None, undefinedValue=(0.0, 0.0)):
		"""

		Note:
			An undefinedValue of None indicates the combined vector field will only be 
			defined on the extents of the component vector fields. If a value is 
			provided, the extents of the combined field will be defined to contain
			all the extents of the components. Typical users will provide a value
			in order to produce a combined vector field with no undefined gaps.
		
		"""
		self._undefinedVal = undefinedValue
		validExtents = extents.NullExtents()

		if (fieldRep is not None):
			self._componentFields = [fieldRep]
			validExtents = fieldRep.validExtents
		else:
			self._componentFields = []


		if (undefinedValue is None):
			# Define combined extents in a piecewise fashion
			self._validExtents = extents.PiecewiseExtents(validExtents)
		else:
			# Define extents which emcompass all component extents
			self._validExtents = extents.EncompassingExtents(validExtents)

	def __getitem__(self, index):
		value = self._undefinedVal

		if (not self._validExtents.contain(index)):
			# point not in valid extents, return undefined value
			return value
		else:
			# point is within at least one component vector field
			if (value is None):
				# initialize None value to (0,0), will be overwritten by component
				value = (0.0, 0.0)

			for field in self._componentFields:
				value = tuple(map(lambda x,y: x + y, value, field[index]))

		return value

	def addField(self, fieldRep):
		# todo: check for mixing piecewise and encompassing exents

		# Check if you are trying to add another compound vector field rep
		if (isinstance(fieldRep, CompoundVectorFieldRepresentation)):
			# add components of other compound vf to self
			self._componentFields.extend(fieldRep.components)

		else:
			self._componentFields.append(fieldRep)

		self._validExtents.addExtents(fieldRep.validExtents)

	@property
	def validExtents(self):
		return self._validExtents

	@property
	def components(self):
		return self._componentFields