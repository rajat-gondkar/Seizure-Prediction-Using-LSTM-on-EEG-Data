#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import relevant libraries and functions for this script
import sys, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
    objArgParse.add_argument('-pl',   '--plot',              required = False, default = True,  help = '')
    objArgParse.add_argument('-rd',   '--resultsdir',        required = False, default = './Results', help = '')

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
    argPlot              = dctArgs['plot'] in (True, 'True', 'true', '1', 1)
    argResultsDir        = dctArgs['resultsdir']
    
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
    force_channels=lstTrainingChannels if lstTrainingChannels else None,
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
        # Check if test data has MORE channels than training (common during CHB-MIT)
        train_set = set(lstTrainingChannels)
        test_set = set(objTestDataset.channels)
        if train_set.issubset(test_set):
            print('  Test data has extra channels — using only the {} training channels'.format(len(lstTrainingChannels)))
        else:
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
lstTestProbs   = []  # Will collect softmax probability vectors

# Move the model to the GPU if one is available
if (blnTrainOnGPU):
    objModelLSTM.cuda()
    
objModelLSTM.eval()

# Wrap test batches in a progress bar
test_iter = objTestLoader
if tqdm:
    test_iter = tqdm(objTestLoader, desc="Test", unit="batch",
                     total=len(objTestLoader), ncols=80)

# Batch loop (each loop feeds one batch of input data)
for arrInputData, arrLabels in test_iter:
    intBatchSizeActual = arrInputData.shape[0]

    # Dynamic hidden state for varying batch sizes (last orphan batch)
    arrHiddenState = objModelLSTM.initHidden(intBatchSizeActual, blnTrainOnGPU, argDebug = False)

    if (blnTrainOnGPU):
        arrInputData, arrLabels = arrInputData.cuda(), arrLabels.cuda()

    # Extract new variables for the hidden and cell states to decouple states
    # from backprop history
    arrHiddenState = tuple([arrState.data for arrState in arrHiddenState])

    # Forward pass
    arrOutput, arrHiddenState = objModelLSTM.forward(arrInputData, arrHiddenState, argDebug = False)

    # Calculate test loss for this batch
    fltTestLoss = objCriterion(arrOutput, arrLabels)

    # Record test loss for this batch
    lstTestLosses.append(fltTestLoss.item())

    # Get class probabilities (for ROC curves) and predictions
    arrProbs = torch.softmax(arrOutput, dim=1)
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

    # Collect probabilities for ROC / PR curves
    lstTestProbs.append(utils.fnTensor2Array(arrProbs, blnTrainOnGPU))

    # Update progress bar
    if tqdm and hasattr(test_iter, 'set_postfix'):
        test_iter.set_postfix(loss=f"{fltTestLoss.item():.4f}",
                              acc=f"{intNumCorrect}/{len(lstTestResults)}")

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


# In[ ]


# ---------------------------------------------------------------------------
# Generate result plots
# ---------------------------------------------------------------------------
if argPlot:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        from sklearn.preprocessing import label_binarize

        strResultsDir = os.path.join(argResultsDir, strTimestamp)
        os.makedirs(strResultsDir, exist_ok=True)
        print('\nGenerating result plots in: {}'.format(strResultsDir))

        lstClassNames = ['interictal', 'preictal', 'ictal']
        y_true = arrTestResults_Sorted[:, 1]
        y_pred = arrTestResults_Sorted[:, 2]

        # 1. Training / validation loss curves (from saved model)
        if lstTrainingStepLosses and lstValidationStepLosses:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(lstTrainingStepLosses, label='Training Loss', alpha=0.7)
            ax.plot(lstValidationStepLosses, label='Validation Loss', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(strResultsDir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print('  ✓ loss_curves.png')

        # 2. Confusion matrix (raw counts)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2],
               xticklabels=lstClassNames, yticklabels=lstClassNames,
               xlabel='Predicted', ylabel='True',
               title='Confusion Matrix')
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.savefig(os.path.join(strResultsDir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  ✓ confusion_matrix.png')

        # 3. Normalized confusion matrix
        cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2],
               xticklabels=lstClassNames, yticklabels=lstClassNames,
               xlabel='Predicted', ylabel='True',
               title='Normalized Confusion Matrix')
        thresh = cm_norm.max() / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, format(cm_norm[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black")
        fig.savefig(os.path.join(strResultsDir, 'confusion_matrix_norm.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  ✓ confusion_matrix_norm.png')

        # 4. ROC curves (One-vs-Rest)
        if lstTestProbs:
            arrAllProbs = np.concatenate(lstTestProbs, axis=0)
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = ['blue', 'green', 'red']
            for i, color in zip(range(3), colors):
                if y_true_bin[:, i].sum() > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], arrAllProbs[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2,
                            label='{} (AUC = {:.2f})'.format(lstClassNames[i], roc_auc))
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves (One-vs-Rest)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(strResultsDir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print('  ✓ roc_curves.png')

        # 5. Per-class precision / recall / F1 bar chart
        precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
        recall    = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        f1        = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)

        x = np.arange(len(lstClassNames))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x,         recall,    width, label='Recall',    alpha=0.8)
        ax.bar(x + width, f1,        width, label='F1-Score',  alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Classification Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(lstClassNames)
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.savefig(os.path.join(strResultsDir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  ✓ per_class_metrics.png')

        print('Done! All plots saved to: {}'.format(strResultsDir))

    except Exception as exc:
        print('\nPlotting skipped due to error: {}'.format(exc))
        print('(Test metrics above are still valid.)')


# In[ ]


if (blnBatchMode):
    # Close the log file and redirect output back to stdout and stderr
    datScriptEnd = utils.fnNow()
    print('Script ended on {}'.format(utils.fnGetDatetime(datScriptEnd)))

    datScriptDuration = datScriptEnd - datScriptStart
    print('datScriptDuration = {}'.format(datScriptDuration))

    objLogFile.close()
    sys.stdout = objStdout
    sys.stderr = objStderr

