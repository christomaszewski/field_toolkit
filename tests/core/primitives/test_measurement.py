from unittest import TestCase

import queue

from field_toolkit.core import primitives

class MeasurementTest(TestCase):

	def test_basic_usage(self):
		point = (5, 10)
		vector = (-1, 6)
		score = 1.235

		unscoredMeasurement = primitives.Measurement(point, vector)
		scoredMeasurement = primitives.Measurement(point, vector, score)

		self.assertTrue(unscoredMeasurement < scoredMeasurement)
		self.assertFalse(scoredMeasurement < unscoredMeasurement)
		self.assertFalse(unscoredMeasurement > scoredMeasurement)
		self.assertTrue(scoredMeasurement > unscoredMeasurement)

		self.assertTrue(scoredMeasurement + unscoredMeasurement == score)

	def test_usage_in_priority_queue(self):
		point = (0, 2)
		vector = (1, 1)

		pq = queue.PriorityQueue()

		for score in range(0,20):
			pq.put((-score, primitives.Measurement(point, vector, score)))

		for score in range(4,8):
			pq.put((-score, primitives.Measurement(point, vector, score)))

		self.assertAlmostEqual(pq.get()[1].score, 19)
		self.assertAlmostEqual(pq.get()[1].score, 18)
