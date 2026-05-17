#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import relevant libraries and functions for this script
import sys, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import libDataIO as dio
import libModelLSTM as LSTM
import libUtils as utils
import libCHBMITDataset as chb

from operator import itemgetter


# In[ ]:


# Jupyter magic commands that should only be run when the code is running
# in Jupyter. Set to blnBatchMode to True when running in batch mode 
blnBatchMode = utils.fnIsBatchMode()

if (blnBatchMode):
    print('Running in BATCH mode...')
    
else:
    print('Running in INTERACTIVE mode...')
    # Automatically reload modules before code execution
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

    # Set plotting style
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


# Set up script and training parameters from command line arguments (batch mode)
# or hard-coded values in the script
if (blnBatchMode):
    
    # BATCH MODE ONLY: This cell will execute in batch mode and parse the relevant
    #                  command line arguments
    
    import argparse

    # Construct argument parser
    objArgParse = argparse.ArgumentParser()

    # Add arguments to the parser
    objArgParse.add_argument('-md',   '--modeldir',          required = True,                   help = '')
    objArgParse.add_argument('-mn',   '--modelname',         required = True,                   help = '')
    objArgParse.add_argument('-tcsv', '--csvpath',           required = True,                   help = '')
    
    objArgParse.add_argument('-rf',   '--resamplingfreq',    required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-du',   '--subseqduration',    required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-ss',   '--stepsizetimepts',   required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-sss',  '--stepsizestates', type = json.loads, required = False, default = '{}', help = '')  # Example use: -sss '{"ictal": 128}'
    objArgParse.add_argument('-sw',   '--subwindowfraction', required = False, default = -1,    help = '')  # Will be over-written from model
    
    objArgParse.add_argument('-smod', '--scalingmode',       required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-smin', '--scaledmin',         required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-smax', '--scaledmax',         required = False, default = 1,     help = '')  # Will be over-written from model
    
    objArgParse.add_argument('-sd',   '--shuffledata',       required = False, default = False, help = '')  # No real need to shuffle for testing
    
    objArgParse.add_argument('-gpu',  '--gpudevice',         required = False, default = -1,    help = '')
    objArgParse.add_argument('-nw',   '--numworkers',        required = False, default = 4,     help = '')
    objArgParse.add_argument('-sa',   '--saveanno',          required = False, default = False, help = '')

    dctArgs = vars(objArgParse.parse_args())

    # Convert parameters extract from arguments to their appropriate date types
    argModelDir          = dctArgs['modeldir']
    argModelName         = dctArgs['modelname']
    argCSVPath           = dctArgs['csvpath']
    
    argResamplingFreq    = int(dctArgs['resamplingfreq'])
    argSubSeqDuration    = int(dctArgs['subseqduration'])
    argStepSizeTimePts   = int(dctArgs['stepsizetimepts'])
    argStepSizeStates    = dctArgs['stepsizestates']
    argStepSizeStates    = {}     # Step size of sliding window for specific segment states (default = {}, use -ssv value)
    argSubWindowFraction = float(dctArgs['subwindowfraction'])
    
    argScalingMode       = int(dctArgs['scalingmode'])
    argScaledMin         = int(dctArgs['scaledmin'])
    argScaledMax         = int(dctArgs['scaledmax'])
    
    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    argShuffleData       = dctArgs['shuffledata']
    argGPUDevice         = int(dctArgs['gpudevice'])
    argNumWorkers        = int(dctArgs['numworkers'])
    argSaveAnno          = dctArgs['saveanno']
    
else:
    # Set all configurable parameters of the script as arguments. After
    # these parameters are set, the entire script can be run in one go

    # Specify the path where the trained models are saved
    argModelDir = './SavedModels/'

    # Specify the model name to use for testing
    argModelName = 'EEGLSTM_CHB-MIT_chb15_Epoch-1_TLoss-0.5327_VLoss-0.6197_20200617-044737.net'  # chb15 (AdamW, smod = 0), epoch = 1 -> hard to tell if there is convergence -> test accuracy from training = 0.82 although test loss = 0.5464. Almost zero ictal prediction, good interictal prediction
    
    # Specify the CSV file that lists the EEG segments to use for testing
    argCSVPath        = './DataCSVs/CHB-MIT/chb15_Test.csv'

    # Specify the resampling frequency and subsequence durations for the
    # EEG segments of the test set (most of the following arguments will
    # be over-written by those from the loaded model if the model contains
    # dctModelProperties{})
    argResamplingFreq    = 128
    argSubSeqDuration    = 5
    argStepSizeTimePts   = -1
    argStepSizeStates    = {}     # Step size of sliding window for specific segment states (default = {}, use -ssv value)
    argSubWindowFraction = 0.3
    
    argScalingMode       = 1
    argScaledMin         = -1
    argScaledMax         = 1
    
    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    # There is really no need to shuffle data when we're feeding in the test set
    argShuffleData       = True  # Shuffle data in DataLoader or not (should not affect test results)

    # Specify which GPU device to use
    argGPUDevice = 0
    argNumWorkers = 4
    
    # Specify whether to save test results to annotation files
    argSaveAnno = True


# In[ ]:


# Generate a timestamp that is unique to this run
strTimestamp = str(utils.fnGenTimestamp())
print('strTimestamp = {}'.format(strTimestamp))


# In[ ]:


# Create a log file only when in batch mode
if (blnBatchMode):
    # Log all output messages to a log file when it is in Batch mode
    strLogDir = './Logs/'  # TODO: Make this into an argument?

    # Create a new directory if it does not exist
    utils.fnOSMakeDir(strLogDir)

    # Saving the original stdout and stderr
    objStdout = sys.stdout
    objStderr = sys.stderr

    strLogFilename = 'runTestLSTM_' + strTimestamp + '_' + argModelName + '.log'
    print('strLogFilename = {}'.format(strLogFilename))

    # Open a new log file
    objLogFile = open(strLogDir + strLogFilename, 'w')

    # Replace stdout and stderr with log file so all print statements will
    # be redirected to the log file from this point on
    sys.stdout = objLogFile
    sys.stderr = objLogFile

    datScriptStart = utils.fnNow()
    print('Script started on {}'.format(utils.fnGetDatetime(datScriptStart)))
    print()


# In[ ]:


# Load a saved LSTM model from the file system

strModelDir = argModelDir
strModelName = argModelName

(intTrainNumChannels, intTrainSeqLen, intTrainNumSegments, objModelLSTM,
 intNumEpochs, intBatchSize, blnShuffleIndices, blnShuffleData, fltLearningRate, intPrintEvery, fltGradClip,
 lstTrainingStepLosses, lstValidationStepLosses, dctModelProperties) = LSTM.fnLoadLSTMModel(strModelDir, strModelName)

# Over-write arguments from the script with values saved in the model
if (dctModelProperties):
    lstTrainingChannels  = dctModelProperties['lstTrainingChannels']
    
    argResamplingFreq    = dctModelProperties['fltResamplingFreq']
    argSubSeqDuration    = dctModelProperties['fltSubSeqDuration']
    argStepSizeTimePts   = dctModelProperties['intStepSizeTimePts']
    argStepSizeStates    = utils.fnFindInDct(dctModelProperties, 'dctStepSizeStates', argStepSizeStates)  # May not exist in some models
    argSubWindowFraction = dctModelProperties['fltSubWindowFraction']

    argScalingMode       = dctModelProperties['intScalingMode']
    argScaledMin         = dctModelProperties['fltScaledMin']
    argScaledMax         = dctModelProperties['fltScaledMax']

    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    tupScalingInfo       = utils.fnFindInDct(dctModelProperties, 'tupScalingInfo', ())  # May not exist in some models
    
    intValPerEpoch       = dctModelProperties['intValPerEpoch']
    
    print('\n  dctModelProperties{} exists in saved model. Over-writting the following arguments with values saved in the model:')
    print()
    print('    argResamplingFreq = {}'.format(argResamplingFreq))
    print('    argSubSeqDuration = {}'.format(argSubSeqDuration))
    print('    argStepSizeTimePts = {}'.format(argStepSizeTimePts))
    print('    argStepSizeStates = {}'.format(argStepSizeStates))
    print('    argSubWindowFraction = {}'.format(argSubWindowFraction))
    print('    argScalingParams = {}'.format(argScalingParams))
    print()
    
    if ('strLogFilename' in dctModelProperties.keys()):
        print('    strLogFilename = {}'.format(dctModelProperties['strLogFilename']))
        print()
    
else:
    lstTrainingChaneels = []
    tupScalingInfo      = ()
    intValPerEpoch      = -1
    
    print('\n  WARNING: dctModelProperties{} not found in saved model. Using arguments specified in this training script')
    print()

print('lstTrainingChannels = {}'.format(lstTrainingChannels))
print('len(tupScalingInfo) = {}'.format(len(tupScalingInfo)))
print('intValPerEpoch = {}'.format(intValPerEpoch))
print()

utils.fnShowMemUsage()
print()


# In[ ]:


# Print out all specified arguments
print('argModelDir = {}'.format(argModelDir))
print('argModelName = {}'.format(argModelName))
print('argCSVPath = {}'.format(argCSVPath))

print('argResamplingFreq = {}'.format(argResamplingFreq))
print('argSubSeqDuration = {}'.format(argSubSeqDuration))
print('argStepSizeTimePts = {}'.format(argStepSizeTimePts))
print('argStepSizeStates = {}'.format(argStepSizeStates))
print('argSubWindowFraction = {}'.format(argSubWindowFraction))

print('argScalingParams = {}'.format(argScalingParams))

if (tupScalingInfo):
    print('  -> Scaling across training and test files')

print('argShuffleData = {}'.format(argShuffleData))

print('argGPUDevice = {}'.format(argGPUDevice))

print('argSaveAnno = {}'.format(argSaveAnno))

print()

utils.fnShowMemUsage()
print()


# In[ ]:


# Plot training loss and validation loss for the entire training if in interactive mode
if (not blnBatchMode):
    utils.fnPlotTrainValLosses(lstTrainingStepLosses, lstValidationStepLosses, intValPerEpoch, argXLim = (), argYLim = ())


# In[ ]:


# Read test data using lazy Dataset
strCSVPath = argCSVPath

fltResamplingFreq    = argResamplingFreq
fltSubSeqDuration    = argSubSeqDuration
intStepSizeTimePts   = argStepSizeTimePts
dctStepSizeStates    = argStepSizeStates
fltSubWindowFraction = argSubWindowFraction
tupScalingParams     = argScalingParams
blnShuffleData       = argShuffleData

objTestDataset = chb.CHBMITDataset(
    csv_path=strCSVPath,
    resampling_freq=fltResamplingFreq,
    subseq_duration=fltSubSeqDuration,
    scaling_params=tupScalingParams,
    scaling_info=tupScalingInfo,
    step_size_time_pts=intStepSizeTimePts,
    step_size_states=dctStepSizeStates,
    sub_window_fraction=fltSubWindowFraction,
    anno_suffix='annotation.txt',
    argInfo=True, argDebug=False)

intLabeledTestNumChannels = objTestDataset.num_channels
intLabeledTestSeqLen      = objTestDataset.subseq_timepts_resampled
intLabeledTestNumSegments = len(objTestDataset)

# Extract metadata arrays for post-test analysis (no data duplication for test set)
lstLabeledTestFilenames     = [e['filename'] for e in objTestDataset.index]
lstLabeledTestSegLabels     = [dio.fnGetSegLabel(e['label']) for e in objTestDataset.index]
lstLabeledTestSegTypes      = [e['label'] for e in objTestDataset.index]
lstLabeledTestSegDurations  = [fltSubSeqDuration] * intLabeledTestNumSegments
lstLabeledTestSamplingFreqs = [fltResamplingFreq] * intLabeledTestNumSegments
lstLabeledTestChannels      = [objTestDataset.channels] * intLabeledTestNumSegments
lstLabeledTestSequences     = [e['sequence'] for e in objTestDataset.index]
lstLabeledTestSubSequences  = [e['subsequence'] for e in objTestDataset.index]
arrLabeledTestStartEndTimesSec = np.array([[e['start_sec'], e['end_sec']] for e in objTestDataset.index], dtype=np.float64)

lstLabeledTestUIDs  = list(range(intLabeledTestNumSegments))
lstLabeledTestUUIDs = list(range(intLabeledTestNumSegments))

# Channel consistency check
if lstTrainingChannels:
    if lstTrainingChannels != objTestDataset.channels:
        print('Training channels:\n  {}'.format(lstTrainingChannels))
        print('Test channels:\n  {}'.format(objTestDataset.channels))
        raise Exception('Training channels do not match test channels!')

print('Test dataset: {} windows, {} channels, {} timepts'.format(
    intLabeledTestNumSegments, intLabeledTestNumChannels, intLabeledTestSeqLen))
print()

utils.fnShowMemUsage()
print()

# GPU check (must happen before DataLoader pin_memory)
blnTrainOnGPU = torch.cuda.is_available()
if blnTrainOnGPU:
    torch.cuda.set_device(argGPUDevice)

# Build DataLoader (test data: NO oversampling, keep natural distribution)
objTestLoader = DataLoader(objTestDataset, batch_size=intBatchSize,
                           shuffle=blnShuffleData, num_workers=argNumWorkers,
                           pin_memory=blnTrainOnGPU)


# In[ ]:


# Check if a GPU is available and if so, set a device to use

intGPUDevice = argGPUDevice

if (blnTrainOnGPU):
    intNumGPUs = torch.cuda.device_count()
    print('Training on GPU ({} available):'.format(intNumGPUs))
    for intGPU in range(intNumGPUs):
        print('  Device {}: {}'.format(intGPU, torch.cuda.get_device_name(intGPU)))
    torch.cuda.set_device(intGPUDevice)
    print('Using GPU #{}'.format(intGPUDevice))
else:
    print('No GPU available, training on CPU')


# In[ ]:


# Define loss criterion

import torch.nn as nn

objCriterion = nn.CrossEntropyLoss()


# In[ ]:


# Evaluate model with test data set and record test losses & prediction accuracy

blnDebug = False
lstTestLosses = []  # Record test losses per batch/step
intNumCorrect = 0   # Number of correctly predicted sequences

# Collect results dynamically (since we no longer exclude orphan batches)
lstTestResults = []  # Will collect (uuid, label, prediction) tuples

# Move the model to the GPU if one is available
if (blnTrainOnGPU):
    objModelLSTM.cuda()
    
objModelLSTM.eval()

# Batch loop (each loop feeds one batch of input data)
for arrInputData, arrLabels in objTestLoader:
    intBatchSizeActual = arrInputData.shape[0]
    print('Feed forwarding new test batch (size={})...'.format(intBatchSizeActual))
    
    # Dynamic hidden state for varying batch sizes (last orphan batch)
    arrHiddenState = objModelLSTM.initHidden(intBatchSizeActual, blnTrainOnGPU, argDebug = False)
        
    if (blnTrainOnGPU):
        arrInputData, arrLabels = arrInputData.cuda(), arrLabels.cuda()
    
    # Extract new variables for the hidden and cell states to decouple states
    # from backprop history
    arrHiddenState = tuple([arrState.data for arrState in arrHiddenState])

    # Forward pass
    arrOutput, arrHiddenState = objModelLSTM.forward(arrInputData, arrHiddenState, argDebug = False)
    
    if (blnDebug):
        print('  arrLabels = {}'.format(arrLabels))
        print('  arrOutput = \n{}'.format(arrOutput))
    
    # Calculate test loss for this batch
    fltTestLoss = objCriterion(arrOutput, arrLabels)
    print('  fltTestLoss = {:.6f} ({})'.format(fltTestLoss, fltTestLoss.type()))
    
    # Record test loss for this batch
    lstTestLosses.append(fltTestLoss.item())
    
    # Convert output scores between classes to predictions
    _, arrPredictions = torch.max(arrOutput, 1)
    
    # Compare the class predictions to the test set labels
    arrCorrect = arrPredictions.eq(arrLabels)
    arrCorrect = utils.fnTensor2Array(arrCorrect, blnTrainOnGPU)
    intNumCorrect += np.sum(arrCorrect)
    
    # Convert tensors back to np.arrays
    arrLabels_np = utils.fnTensor2Array(arrLabels, blnTrainOnGPU)
    arrPredictions_np = utils.fnTensor2Array(arrPredictions, blnTrainOnGPU)
    
    # Generate sequential UUIDs for this batch
    intStartUUID = len(lstTestResults)
    for i in range(intBatchSizeActual):
        lstTestResults.append([intStartUUID + i, int(arrLabels_np[i]), int(arrPredictions_np[i])])
    
    # Clear the GPU cache regularly
    torch.cuda.empty_cache()
    
# Convert results list to array
arrTestResults = np.array(lstTestResults, dtype=int)
arrTestResults_Sorted = arrTestResults[arrTestResults[:, 0].argsort()]  # Sort by UUID
intTestSetSize = arrTestResults_Sorted.shape[0]

# Print the mean test loss for the entire test set
print("Test Loss = {:.4f}".format(np.mean(lstTestLosses)))

# Print the test accuracy over all test data
fltTestAccuracy = intNumCorrect / intTestSetSize
print("Test Accuracy = {}/{} = {:.4f}".format(int(round(intNumCorrect)), intTestSetSize, fltTestAccuracy))


# In[ ]


# Post-testing error analysis
print('arrTestResults_Sorted.shape = {}'.format(arrTestResults_Sorted.shape))
print()

# Generate masks for various metrics
arrFalsePositivesMask = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['interictal'][1],
    arrTestResults_Sorted[:, 2] == dio.dctSegStates['ictal'][1])
arrFalseNegativesMask = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['ictal'][1],
    arrTestResults_Sorted[:, 2] == dio.dctSegStates['interictal'][1])
arrTruePositivesMask  = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['ictal'][1],
    arrTestResults_Sorted[:, 2] == dio.dctSegStates['ictal'][1])
arrTrueNegativesMask  = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['interictal'][1],
    arrTestResults_Sorted[:, 2] == dio.dctSegStates['interictal'][1])

intTestSetSize = arrTestResults_Sorted.shape[0]

intNumFalsePositives = arrTestResults_Sorted[arrFalsePositivesMask].shape[0]
intNumFalseNegatives = arrTestResults_Sorted[arrFalseNegativesMask].shape[0]
intNumTruePositives  = arrTestResults_Sorted[arrTruePositivesMask].shape[0]
intNumTrueNegatives  = arrTestResults_Sorted[arrTrueNegativesMask].shape[0]

intNumCorrect = intTestSetSize - intNumFalsePositives - intNumFalseNegatives

print('intTestSetSize = {}, intNumFalsePositives = {}, intNumFalseNegatives = {}, intNumTruePositives = {}, intNumTrueNegatives = {}'
      .format(intTestSetSize, intNumFalsePositives, intNumFalseNegatives, intNumTruePositives, intNumTrueNegatives))

fltTruePositiveRate, fltTrueNegativeRate = utils.fnCalcPerfMetrics(
    intNumFalsePositives, intNumFalseNegatives, intNumTruePositives, intNumTrueNegatives)

fltTestAccuracy = intNumCorrect / intTestSetSize
print('Test Accuracy = {}/{} = {:.4f}'.format(int(round(intNumCorrect)), intTestSetSize, fltTestAccuracy))
print('fltTruePositiveRate = {:.4f}, fltTrueNegativeRate = {:.4f}'.format(fltTruePositiveRate, fltTrueNegativeRate))
print()

print('False positives (interictal predicted as ictal): {} samples'.format(intNumFalsePositives))
print('False negatives (ictal predicted as interictal): {} samples'.format(intNumFalseNegatives))
print()

if argSaveAnno:
    print('Warning: argSaveAnno not supported with lazy dataset (skipping annotation file generation)')
    print()


# In[ ]:


if (blnBatchMode):
    # Close the log file and redirect output back to stdout and stderr
    datScriptEnd = utils.fnNow()
    print('Script ended on {}'.format(utils.fnGetDatetime(datScriptEnd)))

    datScriptDuration = datScriptEnd - datScriptStart
    print('datScriptDuration = {}'.format(datScriptDuration))

    objLogFile.close()
    sys.stdout = objStdout
    sys.stderr = objStderr

