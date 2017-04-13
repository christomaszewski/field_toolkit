import dill
import os

from context import field_toolkit

from field_toolkit.core.fields import VectorField
from field_toolkit.core.extents import FieldExtents
import field_toolkit.viz.plotting as field_viz

""" Script to generate a flow simulating a fully developed channel flow
	as may be found in an unobstructed river. Plots the scenario and 
	saves it to a file in the output directory
"""

# Make sure output directory exists
outputDir = "../output/"
if not os.path.exists(outputDir):
	os.makedirs(outputDir)

# Scenario Output filename
scenarioName = "single_channel"
fileName = outputDir + scenarioName + ".scenario"

# Scenario Parameters
channelWidth = 100 #meters
vMax = 3.0 #m/s

# Domain Description
xOrigin = 0 #meters
yOrigin = 0 #meters
xDist = channelWidth #meters 
yDist = 50 #meters

# Build Field Extents from Domain Description
domainExtents = FieldExtents.from_bounds_list([xOrigin, xDist, yOrigin, yDist])

# Build Vector Field from developed pipe flow model
flowField = VectorField.from_developed_pipe_flow_model(channelWidth, vMax, domainExtents)

# Dump field to scenario file
with open(fileName, mode='wb') as f:
	dill.dump(flowField, f)

# Read in field from scenario file to demonstrate usage
with open(fileName, mode='rb') as f:
	loadedField = dill.load(f)

# Plot field quiver and pause for 10 seconds
fieldView = field_viz.SimpleFieldView(loadedField, pause=10, autoRefresh=True)
fieldView.quiver()

