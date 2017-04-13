import matplotlib.pyplot as plt
import numpy as np
import dill
import os

from context import field_toolkit

from field_toolkit.core.extents import FieldExtents
import field_toolkit.core.fields as field_lib
import field_toolkit.viz.plotting as field_viz

""" Script to generate a flow simulating a bridge pylon with two structured 
	flows on either side of a region of slow flow, which divereges and then 
	converges around the simulated pylon location (square pylon). Plots the 
	scenario and saves it to the output directory

	Good examples of more advanced extent manipulation and compound vector 
	fields
"""

# Make sure output directory exists
outputDir = "../output/"
if not os.path.exists(outputDir):
	os.makedirs(outputDir)

# Scenario Output filename
scenarioName = "pylon"
fileName = outputDir + scenarioName + ".scenario"

# Scenario Parameters
pylonWidth = 10 #meters
pylonPosition = np.array([65.0, 25.0]) #meters

# Domain Description - Describes entire region to work with
xOrigin = 0 #meters
yOrigin = 0 #meters
xDist = 100 #meters
yDist = 50 #meters

# Build Field Extents from Domain Description
domainExtents = FieldExtents.from_bounds_list([xOrigin, xDist, yOrigin, yDist])

# Define widths of channels making up the flow (meters)
ccWidth = pylonWidth # Create a channel to accomodate the pylon
lcWidth = pylonPosition[0] - pylonWidth / 2 # Define channel to left of pylon
rcWidth = xDist - lcWidth - ccWidth # Define channel to right of pylon

# Compute partition axes (these should be sorted)
partitionAxes = (lcWidth, lcWidth + ccWidth)

# Parition domain extents into channel extents along x axis
lcExtents, ccExtents, rcExtents = domainExtents.xSplit(*partitionAxes)

# Compute partition axes surrounding pylon
pylonStart = pylonPosition - pylonWidth / 2
pylonEnd = pylonPosition + pylonWidth / 2 
pylonPartitionAxes = [pylonStart[1], pylonEnd[1]]

# Partition pylon channel flow extents for pre and post pylon flows
prePylonExtents, pylonExtents, postPylonExtents = ccExtents.ySplit(*pylonPartitionAxes)

# Define a uniform flow across the entire domain
uniformFlow = field_lib.UniformVectorField(flowVector=(-0.2, 0.5), fieldExtents=domainExtents)

# Define a fully developed flow in the left channel
lcVMax = 3.0 #m/s
lcFlow = field_lib.VectorField.from_developed_pipe_flow_model(lcWidth, lcVMax, lcExtents)

# Design center channel flow before and after Pylon
# Faster flow before pylon and a diverging factor to simulate water flowing around the pylon
prePylonFlow = field_lib.UniformVectorField(flowVector=(0.0, 1.0), fieldExtents=prePylonExtents)
divFlow = field_lib.DivergingFlowField(3.0, (pylonPosition[0], 0), prePylonExtents, decay='linear')

# Slower flow behind pylon and converging flow to simulate water flowing back together
postPylonFlow = field_lib.UniformVectorField(flowVector=(0.0, -0.25), fieldExtents=postPylonExtents)
convFlow = field_lib.ConvergingFlowField(2.0, (pylonPosition[0], 0), postPylonExtents, decay='linear')

# Pylon flow to counteract uniform flow component
pylonFlow = field_lib.UniformVectorField(flowVector=(0.2, -0.6), fieldExtents=pylonExtents)

# Combine all center channel flows into single field
ccFlow = field_lib.CompoundVectorField(prePylonFlow, pylonFlow, postPylonFlow, divFlow, convFlow)

# Define a fully developed flow in the right channel
rcVMax = 1.5
xOffset = lcWidth + ccWidth
rcFlow = field_lib.DevelopedPipeFlowField(rcWidth, rcVMax, rcExtents, offset=(xOffset, 0.0))

# Combine flow from left, right, and center channels with uniform flow across the domain
compoundVF = field_lib.CompoundVectorField(uniformFlow, lcFlow, ccFlow, rcFlow)

# Save field to scenario file
with open(fileName, mode='wb') as f:
	dill.dump(compoundVF, f)

# Read field from scenario file to demonstrate usage
with open(fileName, mode='rb') as f:
	loadedField = dill.load(f)

# Plot field quiver and pause for 10 seconds
fieldView = field_viz.SimpleFieldView(loadedField, pause=10, autoRefresh=True)
fieldView.quiver()