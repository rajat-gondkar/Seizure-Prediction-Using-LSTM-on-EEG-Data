#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import relevant libraries and functions for this script
import sys, os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

import libDataIO as dio
import libModelLSTM as LSTM
import libUtils as utils
import libCHBMITDataset as chb

from operator import itemgetter

# Set up for Gmail outgoing email server for script notifications
strGmailSMTPServer = 'smtp.gmail.com'
intGmailSMTPPort = 587


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


# Get more verbose traceback info when an error occurs
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# In[ ]:


# Generate a timestamp that is unique to this run
strTimestamp = str(utils.fnGenTimestamp())
print('strTimestamp = {}'.format(strTimestamp))


# In[ ]:


# Print the script name and arguments for reference if in batch mode
if (blnBatchMode): print(' '.join(sys.argv))


# In[ ]:


# Log all output messages to a log file if in batch mode
if (blnBatchMode):
    
    strLogDir = './Logs/'  # TODO: Make this into an argument?

    # Create a new directory if it does not exist
    utils.fnOSMakeDir(strLogDir)

    # Saving the original stdout and stderr
    objStdout = sys.stdout
    objStderr = sys.stderr

    strLogFilename = 'runTrainLSTM_' + strTimestamp + '.log'
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


# Set up script and training parameters from command line arguments (batch mode)
# or hard-coded values in the script
if (blnBatchMode):
    
    # BATCH MODE ONLY: This cell will execute in batch mode and parse the relevant
    #                  command line arguments
    
    import argparse
    import json

    # Construct argument parser
    objArgParse = argparse.ArgumentParser()

    # Add arguments to the parser
    objArgParse.add_argument('-csv',  '--csvpath',            required = True,                  help = '')
    objArgParse.add_argument('-tcsv', '--testcsvpath',        required = False, default = '',   help = '')
    
    objArgParse.add_argument('-rf',   '--resamplingfreq',     required = False, default = -1,   help = '')
    objArgParse.add_argument('-du',   '--subseqduration',     required = False, default = -1,   help = '')
    objArgParse.add_argument('-ss',   '--stepsizetimepts',    required = False, default = -1,   help = '')
    objArgParse.add_argument('-sss',  '--stepsizestates', type = json.loads, required = False, default = '{}', help = '')  # Example use: -sss '{"ictal": 128}'
    objArgParse.add_argument('-sw',   '--subwindowfraction',  required = False, default = -1,   help = '')
    
    objArgParse.add_argument('-smod', '--scalingmode',        required = False, default = -1,   help = '')
    objArgParse.add_argument('-smin', '--scaledmin',          required = False, default = -1,   help = '')
    objArgParse.add_argument('-smax', '--scaledmax',          required = False, default = 1,    help = '')

    objArgParse.add_argument('-vf',   '--valsetfrac',         required = False, default = 0.2,  help = '')
    objArgParse.add_argument('-tf',   '--testsetfrac',        required = False, default = 0.1,  help = '')
    objArgParse.add_argument('-si',   '--shuffleindices',     required = False, default = True, help = '')
    objArgParse.add_argument('-sd',   '--shuffledata',        required = False, default = True, help = '')
    objArgParse.add_argument('-bs',   '--batchsize',          required = True,                  help = '')

    objArgParse.add_argument('-gpu',  '--gpudevice',          required = False, default = -1,   help = '')
    objArgParse.add_argument('-nw',   '--numworkers',         required = False, default = 4,    help = '')

    objArgParse.add_argument('-hd',   '--hiddendim',          required = True,                  help = '')
    objArgParse.add_argument('-nl',   '--numlayers',          required = True,                  help = '')
    objArgParse.add_argument('-os',   '--outputsize',         required = True,                  help = '')
    objArgParse.add_argument('-dr',   '--dropprob',           required = False, default = 0.5,  help = '')

    objArgParse.add_argument('-cw',   '--classweights', nargs = '+', required = False, default = [], help = '')  # Example use: -cv 1 1 5
    objArgParse.add_argument('-opt',  '--optimizer',          required = False, default = 0,    help = '')
    
    objArgParse.add_argument('-lr',   '--learningrate',       required = True,                  help = '')
    objArgParse.add_argument('-ep',   '--numepochs',          required = True,                  help = '')
    objArgParse.add_argument('-ve',   '--valperepoch',        required = False, default = 20,   help = '')
    objArgParse.add_argument('-gc',   '--gradclip',           required = False, default = 5,    help = '')

    objArgParse.add_argument('-lg',   '--emaillogin',         required = False,                 help = '')
    objArgParse.add_argument('-pw',   '--emailpasswd',        required = False,                 help = '')
    objArgParse.add_argument('-fr',   '--fromemail',          required = False,                 help = '')
    objArgParse.add_argument('-to',   '--toemails',           required = False,                 help = '')

    # Extract the arguments from the command line
    dctArgs = vars(objArgParse.parse_args())

    # Convert parameters extract from arguments to their appropriate date types
    argCSVPath           = dctArgs['csvpath']
    argTestCSVPath       = dctArgs['testcsvpath']
    
    argResamplingFreq    = int(dctArgs['resamplingfreq'])
    argSubSeqDuration    = int(dctArgs['subseqduration'])
    argStepSizeTimePts   = int(dctArgs['stepsizetimepts'])
    argStepSizeStates    = dctArgs['stepsizestates']
    argSubWindowFraction = float(dctArgs['subwindowfraction'])
    
    argScalingMode       = int(dctArgs['scalingmode'])
    argScaledMin         = int(dctArgs['scaledmin'])
    argScaledMax         = int(dctArgs['scaledmax'])
    
    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    argValSetFrac        = float(dctArgs['valsetfrac'])
    argTestSetFrac       = float(dctArgs['testsetfrac'])
    argShuffleIndices    = dctArgs['shuffleindices']
    argShuffleData       = dctArgs['shuffledata']
    argBatchSize         = int(dctArgs['batchsize'])

    argGPUDevice         = int(dctArgs['gpudevice'])
    argNumWorkers        = int(dctArgs['numworkers'])

    #intFeaturesDim      = intTrainNumChannels           # TODO: Find a way to specify which channels to use
    argHiddenDim         = int(dctArgs['hiddendim'])
    argNumLayers         = int(dctArgs['numlayers'])
    argOutputSize        = int(dctArgs['outputsize'])    # TODO: Extract this from training data (2 states: interictal and preictal)
    argDropProb          = float(dctArgs['dropprob'])

    argClassWeights      = np.array(dctArgs['classweights'], dtype = np.float32)
    argOptimizer         = float(dctArgs['optimizer'])
    
    argLearningRate      = float(dctArgs['learningrate'])
    argNumEpochs         = int(dctArgs['numepochs'])     # Number of epochs (entire training set) to train the model
    argValPerEpoch       = int(dctArgs['valperepoch'])   # Number of validation loops per epoch
    argGradClip          = float(dctArgs['gradclip'])    # Value at which gradient is clipped

    argEmailLogin        = dctArgs['emaillogin']         # Login for email account used for notification
    argEmailPasswd       = dctArgs['emailpasswd']        # Passwd for email account used for notification
    argFromEmail         = dctArgs['fromemail']          # Email address for account used for notification
    argToEmails          = dctArgs['toemails']           # Where to send the email notifications (comma separated for multiple emails)
    
    print('Running in BATCH mode. Using arguments from command line:')
    
else:

    # INTERACTIVE MODE ONLY: Run this cell only when the code is being run in
    #                        Jupyter. It will detect that no arguments have
    #                        been parsed from the command line and will apply
    #                        the following values instead
    
    argCSVPath           = './DataCSVs/CHB-MIT/chb01.csv'
    
    #argTestCSVPath       = ''
    argTestCSVPath       = './DataCSVs/CHB-MIT/chb01_Test.csv'
    
    # --- Cloud-optimized defaults (16 GB RAM, 6 GB VRAM) ---
    argResamplingFreq    = 128    # Downsample to 128 Hz (cuts memory ~50%, minimal info loss for seizure detection)
    argSubSeqDuration    = 5      # 5-second windows (fewer samples, more temporal context for LSTM)
    argStepSizeTimePts   = -1     # Step size of sliding window in time points (default = -1, no sliding window)
    argStepSizeStates    = {}     # Step size of sliding window for specific segment states (default = {}, use -ssv value)
    argSubWindowFraction = 0.3    # Fraction of sliding window to use to determine segment type (default = -1, entire window)
    
    argScalingMode       = 1      # Type of scaling to be applied to the preprocessed data
    argScaledMin         = -1     # Minimum value of the scaled data
    argScaledMax         = 1      # Maximum value of the scaled data
    
    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    argValSetFrac        = 0.2    # Fraction of training set to reserve for validation
    argTestSetFrac       = 0.1    # Fraction of training set to reserve for testing
    argShuffleIndices    = True   # Randomly shuffle training set sequence indices prior to training
    argShuffleData       = True   # Randomly shuffle data batches prior to training
    argBatchSize         = 16     # Batch size tuned for ~6 GB VRAM (RTX 4050)

    argGPUDevice         = 0      # Which GPU device to use for training
    argNumWorkers        = 4      # Parallel data-loading workers

    #intFeaturesDim      = intTrainNumChannels  # TODO: Find a way to specify which channels to use
    argHiddenDim         = 256    # Number of dimensions of the LSTM hidden layers
    argNumLayers         = 2      # Number of LSTM layers
    #argOutputSize       = 2      # TODO: Extract this from training data (2 states: interictal and preictal)
    argOutputSize        = 3      # TODO: Extract this from training data (3 states: interictal, preictal, and ictal)
    argDropProb          = 0.5    # Dropout rate (for regularization)

    argClassWeights      = []     # Give weight to each class in loss function
    argOptimizer         = 1      # Which optimizer to use (0 = Adam, 1 = AdamW, 2 = SGD. Default = 0)
    
    argLearningRate      = 0.001  # Learning rate
    argNumEpochs         = 1      # Number of trainging epochs
    argValPerEpoch       = 20     # Number of validation loops per epoch
    argGradClip          = 5      # Value at which gradient is clipped

    argEmailLogin        = 'RodentSys'
    argFromEmail         = 'RodentSys@gmail.com'
    argToEmails          = '4089300606@txt.att.net'
    
    print('Running in INTERACTIVE mode. Using hard-coded values from script:')


# In[ ]:


# Print out all specified arguments
print('argCSVPath = {}'.format(argCSVPath))
print('argTestCSVPath = {}'.format(argTestCSVPath))

print('argResamplingFreq = {}'.format(argResamplingFreq))
print('argSubSeqDuration = {}'.format(argSubSeqDuration))
print('argStepSizeTimePts = {}'.format(argStepSizeTimePts))
print('argStepSizeStates = {}'.format(argStepSizeStates))
print('argSubWindowFraction = {}'.format(argSubWindowFraction))

print('argScalingParams = {}'.format(argScalingParams))

print('argValSetFrac = {}'.format(argValSetFrac))
print('argTestSetFrac = {}'.format(argTestSetFrac))
print('argShuffleIndices = {}'.format(argShuffleIndices))
print('argShuffleData = {}'.format(argShuffleData))
print('argBatchSize = {}'.format(argBatchSize))

print('argGPUDevice = {}'.format(argGPUDevice))
print('argNumWorkers = {}'.format(argNumWorkers))

#print('argFeaturesDim = {}'.format(argFeaturesDim))
print('argHiddenDim = {}'.format(argHiddenDim))
print('argNumLayers = {}'.format(argNumLayers))
print('argOutputSize = {}'.format(argOutputSize))
print('argDropProb = {}'.format(argDropProb))

print('argClassWeights = {}'.format(argClassWeights))
print('argOptimizer = {}'.format(argOptimizer))

print('argLearningRate = {}'.format(argLearningRate))
print('argNumEpochs = {}'.format(argNumEpochs))
print('argValPerEpoch = {}'.format(argValPerEpoch))
print('argGradClip = {}'.format(argGradClip))

#print('argLoadModel = {}'.format(argLoadModel))

print('argEmailLogin = {}'.format(argEmailLogin))
print('argEmailPasswd = {}'.format(argEmailPasswd))
print('argFromEmail = {}'.format(argFromEmail))
print('argToEmails = {}'.format(argToEmails))

print()

utils.fnShowMemUsage()
print()


# In[ ]:


# TODO:
#
#       (1) Evaluate the effect of data randomization
#             (a) Randomized segment indices vs non-randomized
#             (b) Shuffling data in DataLoader vs not shuffing
#       (2) Compare training with single patient data vs multipls patients
#       (3) Compare original sampling rate vs down-sampled rate (implemented)
#       (4) Compare subsequences of various lengths
#       (5) Investigate between segments with different file sizes
#       (6) Train using subsequences from different segments
#       (7) Check that all provided EEG segments are of the same voltage
#           scale and are centered identically. Center and normalize all
#           channels and segments?
#       (8) Find the minimum set of channels that can make good predictions
#       (9) Investigate the effect of using a moving window to create
#           subsequences

# NOTE: There are two points of concern of using RNN/LSTM on the Kaggle
#       prediction data set
#
#       (1) Whether we want to randomize the different training segments
#           (which are numbered in chronological order) during training.
#           Right now we decided not to do that since the point of RNNs
#           is to learn from the history of the signal. However, we don't
#           know how long this window should be in order to get good
#           prediction. The segment length from the data set is 10 mins,
#           so we don't know whether it is OK to break up the segments
#           into 1 min lengths, or whether it is better to chain up
#           multiple segments into something longer than 10 mins
#
#       (2) No test labels are provided for the data set, presumably due
#           to fairness so that everyone is doing a blind test. However,
#           this will be a challange on how to evaluate the prediction
#           accuracy of the model. Perhaps we should split the raw
#           training data into 3 sets for training, validation, and test
#
#       (3) The human EEGs are sampled at 5000Hz, may consider down-sampling
#           to 256 or 500Hz to reduce the data set size, which will also
#           reduce the number of parameters required for the model
#
#           Update: If we do not downsample and use the raw data at 5000Hz,
#                   we will run out of GPU memory if we train more than 2
#                   segments at the same time (or increase batch size to
#                   more than 3, or set the number of hidden dim > 256).
#                   Therefore, it makes sense to be able to downsample to
#                   make room for training in larger batches or wider
#                   layers
#
#       (4) Each segment is provided as 10 mins long. May want to consider
#           breaking each segment up into subsegments (e.g. 1 min segments)
#           depending on whether a shorter sequence length is still effective
#           for training the model
#
#       (5) Segments are related and ordered by their sequence numbers. For
#           example: interictal segments 0001 - 0006 are 10-min segments from
#           the same hour arranged in chronological order. As are segments
#           0007 - 0012, 0013 - 0018, 0019 - 0024, 0025 - 0030, 0031 - 0036,
#           0037 - 0042, and 0043 - 0048. Segments 0049 - 0050 are 10-min
#           segments are adjacent in time. The same applies for the preictal
#           segments
#
#           However, it is not obvious what the time relationship is between
#           the interictal and preictal segments
#
#       (6) Looks like if we train with segments within the samne one-hour
#           period, prediction using other segments within the same period is
#           pretty good in general if the training went well (no over-fitting
#           or under-fitting). However, predictions from other time periods
#           are less stellar (around 50%)
#
#           NOTE: Using the same hyperparameters may train some segments better
#                 than others. Do we need to automate the exploration of
#                 hyperparameter space for optimized training? Can this be an
#                 innovation?
#
#       (7) If we train multiple segments from different time periods in the
#           same training (e.g. train using segments 0001, 0002, 0007, and 0008)
#           the prediction result is lower than if we train with segments from
#           the same time period

# Build lazy-loading Dataset (scans files once for metadata, reads windows on-the-fly)
objFullDataset = chb.CHBMITDataset(
    csv_path=argCSVPath,
    resampling_freq=argResamplingFreq,
    subseq_duration=argSubSeqDuration,
    scaling_params=argScalingParams,
    scaling_info=(),
    step_size_time_pts=argStepSizeTimePts,
    step_size_states=argStepSizeStates,
    sub_window_fraction=argSubWindowFraction,
    anno_suffix='annotation.txt',
    argInfo=True, argDebug=False)

intTotalWindows = len(objFullDataset)
print('Total windows in dataset: {}'.format(intTotalWindows))
objFullDataset.count_by_class()
print()

utils.fnShowMemUsage()
print()

# Keep references needed later for model saving
lstTrainingChannels = objFullDataset.channels
intTrainNumChannels = objFullDataset.num_channels
intTrainSeqLen      = objFullDataset.subseq_timepts_resampled
tupScalingInfo      = ()  # computed internally by Dataset

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
fltValSetFrac = argValSetFrac
fltTestSetFrac = argTestSetFrac
fltTrainSetFrac = 1 - fltValSetFrac - fltTestSetFrac

lstAllIndices = np.arange(intTotalWindows)
if argShuffleIndices:
    np.random.seed(1)
    np.random.shuffle(lstAllIndices)

intTrainEnd = round(intTotalWindows * fltTrainSetFrac)
intValEnd   = intTrainEnd + round(intTotalWindows * fltValSetFrac)

lstTrainIndices = lstAllIndices[:intTrainEnd].tolist()
lstValIndices   = lstAllIndices[intTrainEnd:intValEnd].tolist()
lstTestIndices  = lstAllIndices[intValEnd:].tolist()

print('Training set:   {} samples'.format(len(lstTrainIndices)))
print('Validation set: {} samples'.format(len(lstValIndices)))
print('Test set:       {} samples'.format(len(lstTestIndices)))
print()

objTrainDataset = Subset(objFullDataset, lstTrainIndices)
objValDataset   = Subset(objFullDataset, lstValIndices)
objTestDataset  = Subset(objFullDataset, lstTestIndices)

# ---------------------------------------------------------------------------
# Class-balancing via WeightedRandomSampler (no physical data duplication)
# ---------------------------------------------------------------------------
train_labels = [objFullDataset.index[i]['label'] for i in lstTrainIndices]
class_counts = np.bincount(train_labels)
class_weights = np.zeros_like(class_counts, dtype=np.float64)
nonzero = class_counts > 0
class_weights[nonzero] = 1.0 / class_counts[nonzero]
sample_weights = [class_weights[lbl] for lbl in train_labels]

objTrainSampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(lstTrainIndices),
    replacement=True)

print('Class counts in training set: {}'.format(class_counts))
print('Using WeightedRandomSampler for class balancing (no RAM duplication)')
print()

# ---------------------------------------------------------------------------
# GPU check (must happen before DataLoader pin_memory)
# ---------------------------------------------------------------------------
blnTrainOnGPU = torch.cuda.is_available()
if blnTrainOnGPU:
    torch.cuda.set_device(argGPUDevice)

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
intBatchSize = argBatchSize

objTrainLoader = DataLoader(objTrainDataset, batch_size=intBatchSize,
                            sampler=objTrainSampler, num_workers=argNumWorkers,
                            pin_memory=blnTrainOnGPU)
objValLoader   = DataLoader(objValDataset, batch_size=intBatchSize,
                            shuffle=False, num_workers=argNumWorkers,
                            pin_memory=blnTrainOnGPU)
objTestLoader  = DataLoader(objTestDataset, batch_size=intBatchSize,
                            shuffle=False, num_workers=argNumWorkers,
                            pin_memory=blnTrainOnGPU)

print('DataLoaders created:')
print('  Train: {} batches/epoch (batch_size={})'.format(len(objTrainLoader), intBatchSize))
print('  Val:   {} batches/epoch'.format(len(objValLoader)))
print('  Test:  {} batches'.format(len(objTestLoader)))
print()

utils.fnShowMemUsage()
print()

utils.fnShowMemUsage()
print()


# In[ ]:


# ---------------------------------------------------------------------------
# Quick sanity check: pull one batch and inspect shape
# ---------------------------------------------------------------------------
iterTrainLoader = iter(objTrainLoader)
arrTrainDataBatch, arrTrainLabelsBatch = next(iterTrainLoader)

print('arrTrainDataBatch.shape = {}, arrTrainLabelsBatch = {}'.format(
    arrTrainDataBatch.shape, arrTrainLabelsBatch.shape))
print('arrTrainLabelsBatch = {}'.format(arrTrainLabelsBatch))
print()


# In[ ]:


# Report GPU status

intGPUDevice = argGPUDevice

if(blnTrainOnGPU):
    intNumGPUs = torch.cuda.device_count()
    print('Training on GPU ({} available):'.format(intNumGPUs))
    for intGPU in range(intNumGPUs):
        print('  Device {}: {}'.format(intGPU, torch.cuda.get_device_name(intGPU)))
    print('Using GPU #{}'.format(intGPUDevice))
else:
    print('No GPU available, training on CPU')
    
print()


# In[ ]:


# Instantiate the model with hyperparameters
intFeaturesDim = intTrainNumChannels  # TODO: Need to mnake this customizable to use a subset of channels
intHiddenDim   = argHiddenDim
intNumLayers   = argNumLayers
intOutputSize  = argOutputSize        # TODO: Extract this from training data (2 states: interictal and preictal)
fltDropProb    = argDropProb

print('intFeaturesDim = {}, intHiddenDim = {}, intNumLayers = {}, intOutputSize = {}, fltDropProb = {}'.format(intFeaturesDim, intHiddenDim, intNumLayers, intOutputSize, fltDropProb))
print()

objModelLSTM = LSTM.clsLSTM(intFeaturesDim, intHiddenDim, intNumLayers, intOutputSize, argDropProb = fltDropProb)
print(objModelLSTM)
print()

objModelLSTM.showParams()
print()

# Print the total number of parameters in the model
intTotalParams = sum(objParam.numel() for objParam in objModelLSTM.parameters() if objParam.requires_grad)
print('intTotalParams = {} ({})'.format(intTotalParams, type(intTotalParams)))
print()

# Print the size of the dataset
intTrainNumSegments = len(objTrainLoader.dataset)
print('Dataset: {} windows x {} timepts x {} channels = {} values'.format(
    intTrainNumSegments, intTrainSeqLen, intTrainNumChannels,
    intTrainNumSegments * intTrainSeqLen * intTrainNumChannels))
print()


# In[ ]:


# Define loss criterion and optimization function

import torch.nn as nn

fltLearningRate = argLearningRate

# If argClassWeights[] is empty, assign equal weight to each class
if (argClassWeights is None or len(argClassWeights) == 0):
    argClassWeights = [1] * argOutputSize  # Assign equal weight to each class

argClassWeights = np.array(argClassWeights, dtype = np.float32)
print('argClassWeights = {}'.format(argClassWeights))

arrClassWeights = torch.tensor(argClassWeights)
if (blnTrainOnGPU):
    arrClassWeights = arrClassWeights.cuda()
objCriterion = nn.CrossEntropyLoss(weight = arrClassWeights)

if (argOptimizer == 0):
    strOptimizer = 'Adam'
    objOptimizer = torch.optim.Adam(objModelLSTM.parameters(), lr = fltLearningRate)
elif (argOptimizer == 1):
    strOptimizer = 'AdamW'
    objOptimizer = torch.optim.AdamW(objModelLSTM.parameters(), lr = fltLearningRate)
elif (argOptimizer == 2):
    strOptimizer = 'SGD'
    objOptimizer = torch.optim.SGD(objModelLSTM.parameters(), lr = fltLearningRate)
else:
    strOptimizer = 'Adam'
    objOptimizer = torch.optim.Adam(objModelLSTM.parameters(), lr = fltLearningRate)
    
print('strOptimizer = {}, fltLearningRate = {}'.format(strOptimizer, fltLearningRate))


# In[ ]:


# Start the training process

# Define training parameters
blnDebug = False

intNumEpochs   = argNumEpochs    # Number of epochs (entire training set) to train the model
intValPerEpoch = argValPerEpoch  # Number of validation loops per epoch
fltGradClip    = argGradClip     # Value at which gradient is clipped

# Calculate validation loss every n training batches/steps
intNumBatchLoops = len(objTrainLoader)
intPrintEvery = max(1, intNumBatchLoops // intValPerEpoch)

print('intNumEpochs = {}, intValPerEpoch = {}, intPrintEvery = {}, fltGradClip = {}'.format(intNumEpochs, intValPerEpoch, intPrintEvery, fltGradClip))
print()

intBatchLoopIdx = 0  # Loop index for training model with batches of data
lstTrainingStepLosses, lstValidationStepLosses = [], []  # List of losses every n training batches/steps

# Move the model to the GPU if one is available
if(blnTrainOnGPU):
    objModelLSTM.cuda()

objModelLSTM.train()

datTrainingStart = utils.fnNow()
print('Training started on {}'.format(utils.fnGetDatetime(datTrainingStart)))

# Train for a specified number of epochs
for intEpoch in range(intNumEpochs):
    print('intEpoch = {}'.format(intEpoch + 1))
    
    # Batch loop (each loop trains one batch of input data)
    for arrInputData, arrLabels in objTrainLoader:
        print('  intBatchLoopIdx = {}.{} (batch size = {})'.format(intEpoch + 1, intBatchLoopIdx + 1, arrInputData.shape[0]))
        intBatchLoopIdx += 1  # New batch/training loop

        # Dynamic hidden-state init: if batch size changed (last orphan batch),
        # create a new hidden state of the correct size instead of skipping data
        if arrInputData.shape[0] != intBatchSize:
            arrHiddenState = objModelLSTM.initHidden(arrInputData.shape[0], blnTrainOnGPU, argDebug = False)
        else:
            arrHiddenState = objModelLSTM.initHidden(intBatchSize, blnTrainOnGPU, argDebug = False)
        
        if(blnTrainOnGPU):
            arrInputData, arrLabels = arrInputData.cuda(), arrLabels.cuda()

        # Extract new variables for the hidden and cell states to decouple states
        # from backprop history. Otherwise the gradient will be backpropagated
        # through the entire training history
        arrHiddenState = tuple([arrState.data for arrState in arrHiddenState])  # Getting only the data portion
                                                                                # of the hidden/cell states detaches
                                                                                # them from the backprop history

        # Zero the accumulated gradients
        objModelLSTM.zero_grad()

        if (blnDebug):
            print('    arrInputData.shape = {}, arrInputData.type() = {}'.format(arrInputData.shape, arrInputData.type()))
            print('    arrLabels.shape = {}, arrLabels.type() = {}'.format(arrLabels.shape, arrLabels.type()))
            print('    arrHiddenState.shape = ({}, {}), arrHiddenState.type() = ({}, {})'.format(arrHiddenState[0].shape, arrHiddenState[1].shape, arrHiddenState[0].type(), arrHiddenState[1].type()))
            print('    arrLabels = {}'.format(arrLabels))
            
        # Forward pass through the model and get the next hidden state and output
        # Output shape = (batch_size, 1), h shape = (n_layers, batch_size, hidden_dim)
        arrOutput, arrHiddenState = objModelLSTM.forward(arrInputData, arrHiddenState, argDebug = False)

        if (blnDebug):
            print('    arrOutput.shape = {}, arrOutput.type() = {}'.format(arrOutput.shape, arrOutput.type()))
            print('    arrHiddenState.shape = ({}, {}), arrHiddenState.type() = ({}, {})'.format(arrHiddenState[0].shape, arrHiddenState[1].shape, arrHiddenState[0].type(), arrHiddenState[1].type()))
            print('    arrOutput = \n{}'.format(arrOutput))
        
        # Calculate the loss and perform backprop (looks like output shape is
        # not changed after the squeeze)
        # NOTE: arrOutput[] is returned as float since the values are close to
        #       (but not exactly) 0 or 1. However, arrLabels[] is expected to
        #       be of type long)
        fltTrainingLoss = objCriterion(arrOutput, arrLabels)  # arrOutput[] = float, arrLabels[] = long
        print('    fltTrainingLoss = {:.6f} ({})'.format(fltTrainingLoss, fltTrainingLoss.type()))
        
        fltTrainingLoss.backward()
        if (blnDebug):
            print('    arrOutput.squeeze().shape = {}'.format(arrOutput.squeeze().shape))
        
        # Using clip_grad_norm() helps prevent the exploding gradient
        # problem in RNNs / LSTMs
        nn.utils.clip_grad_norm_(objModelLSTM.parameters(), fltGradClip)
        objOptimizer.step()

        # Calculate loss statistics
        if (intBatchLoopIdx % intPrintEvery == 0):
            print('  Calculating loss statistics...')
            
            # Get validation loss
            lstValidationBatchLosses = []
            
            objModelLSTM.eval()
            
            for arrInputData, arrLabels in objValLoader:
                print('    In batch loop... (batch size = {})'.format(arrInputData.shape[0]))
                
                # Dynamic hidden state for varying batch sizes (last orphan batch)
                arrValHiddenState = objModelLSTM.initHidden(arrInputData.shape[0], blnTrainOnGPU, argDebug = False)
                
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                arrValHiddenState = tuple([arrState.data for arrState in arrValHiddenState])

                if(blnTrainOnGPU):
                    arrInputData, arrLabels = arrInputData.cuda(), arrLabels.cuda()

                arrOutput, arrValHiddenState = objModelLSTM.forward(arrInputData, arrValHiddenState)
                
                fltValidationLoss = objCriterion(arrOutput, arrLabels)
                print('    fltValidationLoss = {:.6f} ({})'.format(fltValidationLoss, fltValidationLoss.type()))

                lstValidationBatchLosses.append(fltValidationLoss.item())

            objModelLSTM.train()
            
            # Clear the GPU cache regularly to avoid the following CUDA error:
            #
            #   RuntimeError: CUDA out of memory. Tried to allocate 6.61 GiB 
            #   (GPU 1; 10.76 GiB total capacity; 1.25 GiB already allocated; 
            #   2.39 GiB free; 6.38 GiB cached)
            torch.cuda.empty_cache()
            
            print('  Epoch: {}/{}...'.format(intEpoch + 1, intNumEpochs),
                  'Step: {}...'.format(intBatchLoopIdx),
                  'Training Loss: {:.6f}...'.format(fltTrainingLoss.item()),
                  'Validation Loss: {:.6f}'.format(np.mean(lstValidationBatchLosses)))
            
            lstTrainingStepLosses.append(fltTrainingLoss.item())
            lstValidationStepLosses.append(np.mean(lstValidationBatchLosses))
            
print()

datTrainingEnd = utils.fnNow()
print('Training ended on {}'.format(utils.fnGetDatetime(datTrainingEnd)))

datTrainingDuration = datTrainingEnd - datTrainingStart
print('datTrainingDuration = {}'.format(datTrainingDuration))

print()


# In[ ]:


# Plot training loss and validation loss for the entire training
if (not blnBatchMode):
    utils.fnPlotTrainValLosses(lstTrainingStepLosses, lstValidationStepLosses)


# In[ ]:


# Save the trained model to the file system

strModelDir = './SavedModels/'  # TODO: Make this into an argument?
strDataSetName, strCSVName = dio.fnGetDataSetInfo(argCSVPath)

strModelName = 'EEGLSTM_' + strDataSetName +                '_' + strCSVName +                '_Epoch-' + str(intNumEpochs) +                '_TLoss-' + str('{0:.4f}'.format(lstTrainingStepLosses[-1])) +                '_VLoss-' + str('{0:.4f}'.format(lstValidationStepLosses[-1])) +                '_' + strTimestamp +                '.net'

# Create a new directory if it does not exist
utils.fnOSMakeDir(strModelDir)

# Collect all training and model-specific variables and passing them into the model as
# variable named arguments (**kwargs). This is to make the model more self-contained, and
# in this way there is no need to keep changing fnSaveLSTMModel() as we decide to include
# more information in the saved model
dctModelProperties = {
    'strDataSetName':       strDataSetName,
    'strCSVName':           strCSVName,
    'strModelName':         strModelName,
    
    'lstTrainingFiles':     sorted(set(objFullDataset.files)),
    'lstTrainingChannels':  lstTrainingChannels,  # TODO: Assumes that the list of channels are identical across all data subsequences
    
    'fltResamplingFreq':    argResamplingFreq,
    'fltSubSeqDuration':    argSubSeqDuration,
    'intStepSizeTimePts':   argStepSizeTimePts,
    'dctStepSizeStates':    argStepSizeStates,
    'fltSubWindowFraction': argSubWindowFraction,
    
    'intScalingMode':       argScalingMode,
    'fltScaledMin':         argScaledMin,
    'fltScaledMax':         argScaledMax,
    'tupScalingInfo':       tupScalingInfo,
    
    'intImbalFactor':       0,  # 0 = we used WeightedRandomSampler instead of physical oversampling
    
    'fltValSetFrac':        argValSetFrac,
    'fltTestSetFrac':       argTestSetFrac,
    'blnShuffleIndices':    argShuffleIndices,       # Already passed in as positional argument (not removed for backwards-compatibility)
    'blnShuffleData':       argShuffleData,          # Already passed in as positional argument (not removed for backwards-compatibility)
    'intBatchSize':         argBatchSize,            # Already passed in as positional argument (not removed for backwards-compatibility)
    'arrClassWeights':      argClassWeights,
    'intOptimizer':         argOptimizer,
    'fltLearningRate':      argLearningRate,         # Already passed in as positional argument (not removed for backwards-compatibility)
    'intNumEpochs':         argNumEpochs,            # Already passed in as positional argument (not removed for backwards-compatibility)
    'intValPerEpoch':       argValPerEpoch,
    'intPrintEvery':        intPrintEvery,           # Already passed in as positional argument (not removed for backwards-compatibility)
    'fltGradClip':          argGradClip              # Already passed in as positional argument (not removed for backwards-compatibility)
}

if (blnBatchMode):
    dctModelProperties['strLogFilename'] = strLogFilename  # Record log file only if it exists (when trained in batch mode)

LSTM.fnSaveLSTMModel(strModelDir, strModelName, intTrainNumChannels, intTrainSeqLen, intTrainNumSegments,
                     objModelLSTM, intNumEpochs, intBatchSize, argShuffleIndices, argShuffleData, fltLearningRate, intPrintEvery, fltGradClip,
                     lstTrainingStepLosses, lstValidationStepLosses, **dctModelProperties)

print()


# In[ ]:


# Evaluate model with test data set and record test losses & prediction accuracy

blnDebug = True
lstTestLosses = []  # Record test losses per batch/step
intNumCorrect = 0   # Number of correctly predicted sequences in a batch size of (intBatchSize)

# Move the model to the GPU if one is available
if(blnTrainOnGPU):
    objModelLSTM.cuda()

objModelLSTM.eval()

# Batch loop (each loop trains one batch of input data)
for arrInputData, arrLabels in objTestLoader:
    print('Feed forwarding new test batch...')

    # Dynamic hidden state for varying batch sizes (last orphan batch)
    arrHiddenState = objModelLSTM.initHidden(arrInputData.shape[0], blnTrainOnGPU, argDebug = False)
                
    if(blnTrainOnGPU):
        arrInputData, arrLabels = arrInputData.cuda(), arrLabels.cuda()

    # Extract new variables for the hidden and cell states to decouple states
    # from backprop history. Otherwise the gradient will be backpropagated
    # through the entire training history
    arrHiddenState = tuple([arrState.data for arrState in arrHiddenState])

    # Forward pass through the model and get the next hidden state and output
    # Output shape = (batch_size, 1), h shape = (n_layers, batch_size, hidden_dim)
    arrOutput, arrHiddenState = objModelLSTM.forward(arrInputData, arrHiddenState)

    if (blnDebug):
        #print('  arrLabels.shape = {}, arrLabels.type() = {}'.format(arrLabels.shape, arrLabels.type()))
        #print('  arrOutput.shape = {}, arrOutput.type() = {}'.format(arrOutput.shape, arrOutput.type()))
        print('  arrLabels = {}'.format(arrLabels))
        print('  arrOutput = \n{}'.format(arrOutput))

    # Calculate test loss for this batch
    fltTestLoss = objCriterion(arrOutput, arrLabels)
    print('  fltTestLoss = {:.6f} ({})'.format(fltTestLoss, fltTestLoss.type()))

    # Record test loss for this batch
    lstTestLosses.append(fltTestLoss.item())

    # Convert output scores between classes to one-hot encoding
    # that indicate the predictions for each batch
    _, arrPredictions = torch.max(arrOutput, 1)

    # Compare the class predictions to the test set labels
    arrCorrect = arrPredictions.eq(arrLabels)
    if (blnTrainOnGPU):
        arrCorrect = arrCorrect.cpu().numpy()  # Convert tensor back to np.array
    else:
        arrCorrect = arrCorrect.numpy()  # Convert tensor back to np.array
    intNumCorrect += np.sum(arrCorrect)  # Count the number of correctly predicted sequences

    # Clear the GPU cache regularly to avoid the following CUDA error:
    #
    #   RuntimeError: CUDA out of memory. Tried to allocate 6.61 GiB 
    #   (GPU 1; 10.76 GiB total capacity; 1.25 GiB already allocated; 
    #   2.39 GiB free; 6.38 GiB cached)
    torch.cuda.empty_cache()

# Print the mean test loss for the entire test set
print("Test Loss = {:.4f}".format(np.mean(lstTestLosses)))

# Print the test accuracy over all test data
fltTestAccuracy = intNumCorrect / len(objTestLoader.dataset)
print("Test Accuracy = {}/{} = {:.4f}".format(int(round(intNumCorrect)), len(objTestLoader.dataset), fltTestAccuracy))


# In[ ]:


# Close the log file and redirect output back to stdout and stderr
if (blnBatchMode):
    
    datScriptEnd = utils.fnNow()
    print('Script ended on {}'.format(utils.fnGetDatetime(datScriptEnd)))

    datScriptDuration = datScriptEnd - datScriptStart
    print('datScriptDuration = {}'.format(datScriptDuration))

    objLogFile.close()
    sys.stdout = objStdout
    sys.stderr = objStderr


# In[ ]:


# Send email notification to the specified email address(es) to notify
# the end of the script if in batch mode
if (blnBatchMode):
    
    strEmailLogin  = argEmailLogin
    strEmailPasswd = argEmailPasswd
    strFromEmail   = argFromEmail
    strToEmails    = argToEmails

    print('Sending notification to:')
    print('  strEmailLogin = {}'.format(strEmailLogin))
    print('  strFromEmail = {}'.format(strFromEmail))
    print('  strToEmails = {}'.format(strToEmails))

    if strEmailLogin and strFromEmail and strToEmails:
        strSubject = 'Script execution ended (scrTrainLSTM.py)'
        strBody = 'scrTrainLSTM.py finished execution\nDuration = {}\nLog file = {}'.format(datScriptDuration, strLogFilename)
        utils.fnSendMail(strGmailSMTPServer, intGmailSMTPPort, strEmailLogin, strEmailPasswd, strFromEmail, strToEmails, strSubject, strBody)
    else:
        print('  (skipped – no email credentials provided)')


# In[ ]:


print()

