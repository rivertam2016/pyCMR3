# pyCMR3

This project hosts the code developed for: 

Cohen & Kahana (in prep). Retrieved-context theory of memory in emotional disorders.

It contains three directories: 

1. Core_Model
2. Data
3. ParticleSwarm

# Core_Model

Core_Model contains the CMR3 code used in Cohen & Kahana (in prep). This
code can be set to run a CMR2 version by setting beta_source equal to 0.0.
It can be set to run a version of eCMR (Talmi et al., 2019, Psychological Review) 
by setting the number of sources in the model equal to 1. Note however that
this version of eCMR differs from the eCMR version in that paper in a couple
ways (carries learning across multiple lists, simulates a distractor task).

Examples of both these adaptations are provided in the code.

The CMR3 code is written in cython, which is a hybrid language that combines
features of c and python. As such, the model code CMR3.pyx will need to 
be compiled before it can be used. Once it has been compiled, CMR3 can
be imported and used like any other python package.

To build and compile the CMR3 code, navigate to the Core_Model directory
and run the following from the command line:

>> python3 setup_cmr3.py build_ext --inplace

Note that this code is written for compatibility with Python3. It is not tested 
or debugged for use with Python2.

# Data

Data contains 10 example sets of data files for 10 of the subjects 
used in Cohen & Kahana (in prep). The full set of data files, for all 97
subjects included that paper, can be found here, under the title for the
the Cohen & Kahana paper under the "Submitted" papers heading:

http://memory.psych.upenn.edu/Publications

Each subject has three associated files, each located in one of three
directories: pres_files, rec_files, and eval_files. 

1. pres_files/  contains the lists of items that were presented to each
subject. 

2. rec_files/  contains the lists of items that were recalled
by each subject after studying each list in their pres_nos_LTP***.txt file in pres_files.

3. eval_files/  contains files that have the emotional valence codes 
that correspond to each subject's presented-items.

# Particle Swarm

This directory contains an example of how to use particle swarm code to 
obtain best-fitting parameters for the CMR3 model when fitting it to
graphs from behavioral analyses. 

This code uses the pyswarm package for particle swarm optimization, which
I have adapted so that it can be distributed over cores on a cluster using
the lock files method. 

The example code shows how to fit the serial position curve (SPC) 
and the probability of first recall (PFR).
