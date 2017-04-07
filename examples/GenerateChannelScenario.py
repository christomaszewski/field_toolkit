import dill
import os

from context import field_toolkit

import field_toolkit.core.fields as field_lib
import field_toolkit.core.extents as field_extents
import field_toolkit.viz.plotting as field_viz


# Make sure output directory exists
outputDir = "../output/"
if not os.path.exists(outputDir):
	os.makedirs(outputDir)

# Scenario Output filename
scenarioName = "single_channel"
fileName = outputDir + scenarioName + ".scenario"

# Scenario Parameters
channelWidth = 100
vMax = 3.0

# Domain Description
xOrigin = 0 #meters
yOrigin = 0 #meters
xDist = channelWidth #meters 
yDist = 50 #meters

# Build Field Extents from Domain Description
domainExtents = field_extents.FieldExtents.from_list([xOrigin, xDist, yOrigin, yDist])

channelFlowField = field_lib.DevelopedPipeFlowField(channelWidth, vMax, domainExtents)

fieldView = field_viz.SimpleFieldView(channelFlowField, pause=10, autoRefresh=True)
fieldView.quiver()

with open(fileName, mode='wb') as f:
	dill.dump(channelFlowField, f)