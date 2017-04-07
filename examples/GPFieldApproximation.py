import numpy as np
import dill

from context import field_toolkit

import field_toolkit.core.fields as field_lib
import field_toolkit.approx as field_approx
import field_toolkit.viz.plotting as field_viz

""" Script to show how to use GP regression to reconstruct a vector field from samples

	Loads a scenario from file, samples a random points within the field, and attempts to 
	reconstruct the source field from these samples using GP regression
"""

# Load Scenario
scenarioName = 'single_channel'

# Scenario Field file name
scenarioFile = '../output/' + scenarioName + '.scenario'

# Todo:Add check for file
with open(scenarioFile, mode='rb') as f:
	vfSource = dill.load(f)

# Setup field views for plotting both source and reconstructed fields
sourceFieldView = field_viz.SimpleFieldView(vfSource, pause=5)
approxFieldView = field_viz.SimpleFieldView(pause=5, autoRefresh=True)

# Define number of random samples to take from source field
nSamples = 100

# Take random samples from source field
# Assumes corner in origin, need to add random sampling into field or extent objects
randomPoint = lambda : tuple(np.random.rand(2) * list(vfSource.extents.size))
points = [randomPoint() for _ in np.arange(nSamples)]
measurements = list(vfSource.measureAtPoints(points))

# Initialize GP Approximator
vfApproximator = field_approx.gp.GPApproximator()

# Process measurements taken at random points
vfApproximator.addMeasurements(measurements)
vfApprox = vfApproximator.approximate()

# Since we know the source field we can resuse the known extents
vfApprox.extents = vfSource.extents

# Plot source and reconstruction
sourceFieldView.quiver()
approxFieldView.changeField(vfApprox)

sourceFieldView.save('../output/source.png')
approxFieldView.save('../output/approx.png')