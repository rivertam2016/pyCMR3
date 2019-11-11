###########
#
#   Simulate data for an individual subject
#
###########

1.  Build the CMR code:

>> python3 setup_cmr3.py build_ext --inplace

2.  Open run_CMR3.py.

In the main function, at line 111, select the subject you would like to simulate
data for.

3.  At lines 118-120, select which version of the model you would like to run.

4.  Run the code to generate model-predicted recall outputs for each presented list.

###########
#
#   Graph the model-predicted output
#
###########

1.  Open graph_CMR3.py

2.  Select which subject's data to analyze and graph the output for.

In the main function, at line 340, select which subject ID to analyze and
graph the output for.  All subjects' ID's are in the format 'LTP***', where
each * is an integer. See '/CohenKahana2019/Data/valid_subjects_list.txt'
for a list of virtual subjects whose data was valid for inclusion in the
CMR3 paper (Cohen & Kahana, in prep).

3.  Select which model version you want to run.

Set the appropriate model version at lines 349-351 equal to True.

4.  For aggregate analyses, as in Simulation 1:

To graph the aggregate results across each model-simulated dataset,
follow the procedures listed in Sim1_EmotionalClusteringEffect,
in the Graphing_Files directory.