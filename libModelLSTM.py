#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torch.nn as nn


# In[ ]:


'''
Create an LSTM model that will be used to analyze multichannel EEG signal
'''

class clsLSTM(nn.Module):
    def __init__(self, argFeaturesDim, argHiddenDim, argNumLayers, argOutputSize, argDropProb = 0.5, argDebug = False):
        super(clsLSTM, self).__init__()
        
        self.intFeaturesDim = argFeaturesDim
        self.intHiddenDim   = argHiddenDim
        self.intNumLayers   = argNumLayers
        self.intOutputSize  = argOutputSize
        self.fltDropProb    = argDropProb
        self.blnBidirectional = True
        self.intLSTMOutputDim = argHiddenDim * 2 if self.blnBidirectional else argHiddenDim
        
        # Bidirectional LSTM for better temporal context
        self.LSTMLayer = nn.LSTM(argFeaturesDim, argHiddenDim, argNumLayers,
                                 dropout=argDropProb, batch_first=True,
                                 bidirectional=self.blnBidirectional)
        
        # Attention mechanism: learns which time steps matter most
        self.AttentionLayer = nn.Sequential(
            nn.Linear(self.intLSTMOutputDim, argHiddenDim),
            nn.Tanh(),
            nn.Linear(argHiddenDim, 1)
        )
        
        self.DropoutLayer = nn.Dropout(p=argDropProb)
        self.FCLayer = nn.Linear(self.intLSTMOutputDim, argOutputSize)
        
        
    # Display all the named parameters and their shapes in the model
    def showParams(self):
        intParamIdx = 0
        
        for tupParam in self.named_parameters():
            print('{}: {} -> {}'.format(intParamIdx, tupParam[0], tupParam[1].data.shape))
            intParamIdx = intParamIdx + 1            
            
            
    # Perform a forward pass on the model provided with input data and a previous
    # hidden state
    def forward(self, argDataIn, argHiddenIn, argDebug = False):        
        intBatchSize = argDataIn.shape[0]
        
        # Feed input through bidirectional LSTM
        arrLSTMOut, arrHiddenOut = self.LSTMLayer(argDataIn.float(), argHiddenIn)
        if (argDebug): print('arrLSTMOut.shape = {}'.format(arrLSTMOut.shape))
        
        # Attention: compute weights over time steps
        attn_scores = self.AttentionLayer(arrLSTMOut)          # (batch, time, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)      # (batch, time, 1)
        arrContext = torch.sum(arrLSTMOut * attn_weights, dim=1)  # (batch, lstm_output_dim)
        if (argDebug): print('arrContext.shape = {}'.format(arrContext.shape))
        
        # Dropout on the context vector (not on every time step)
        arrDropoutOut = self.DropoutLayer(arrContext)
        if (argDebug): print('arrDropoutOut.shape = {}'.format(arrDropoutOut.shape))
        
        # FC layer for classification
        arrOutput = self.FCLayer(arrDropoutOut)
        if (argDebug): print('arrOutput.shape = {}'.format(arrOutput.shape))
                
        return arrOutput, arrHiddenOut
    
    
    # Initialize the hidden and cell states with zeros
    def initHidden(self, argBatchSize, argTrainOnGPU = False, argDebug = False):
        # For bidirectional LSTM, num_directions=2; hidden state shape:
        # (num_layers * num_directions, batch_size, hidden_dim)
        intNumDirections = 2 if self.blnBidirectional else 1
        intHiddenLayers = self.intNumLayers * intNumDirections
        
        arrWeight = next(self.parameters()).data
        
        if (argTrainOnGPU):
            arrHiddenState = (arrWeight.new(intHiddenLayers, argBatchSize, self.intHiddenDim).zero_().cuda(),
                              arrWeight.new(intHiddenLayers, argBatchSize, self.intHiddenDim).zero_().cuda())
        else:
            arrHiddenState = (arrWeight.new(intHiddenLayers, argBatchSize, self.intHiddenDim).zero_(),
                              arrWeight.new(intHiddenLayers, argBatchSize, self.intHiddenDim).zero_())
        
        if (argDebug):
            print('intNumLayers = {}, intNumDirections = {}, argBatchSize = {}, intHiddenDim = {}'.format(
                self.intNumLayers, intNumDirections, argBatchSize, self.intHiddenDim))
            print('arrHiddenState.shape = ({}, {})'.format(arrHiddenState[0].shape, arrHiddenState[1].shape))
            
        return arrHiddenState


# In[ ]:


def fnSaveLSTMModel(argModelDir, argModelName, argNumChannels, argSeqLen, argNumSegments, argModel, 
                    argNumEpochs, argBatchSize, argShuffleIndices, argShuffleData, argLearningRate, argPrintEvery, argGradClip, 
                    argTrainingStepLosses, argValidationStepLosses, argDebug = False, **argModelProperties):
    print('Saving model: argModelName = {}'.format(argModelName))
        
    dctModelCheckPt = {
        'intNumChannels':          argNumChannels,
        'intSeqLen':               argSeqLen,
        'intNumSegments':          argNumSegments,

        'intFeaturesDim':          argModel.intFeaturesDim,
        'intHiddenDim':            argModel.intHiddenDim,
        'intNumLayers':            argModel.intNumLayers,
        'intOutputSize':           argModel.intOutputSize,
        'fltDropProb':             argModel.fltDropProb,
        'dctStateDict':            argModel.state_dict(),

        'intNumEpochs':            argNumEpochs,        # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'intBatchSize':            argBatchSize,        # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'blnShuffleIndices':       argShuffleIndices,   # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'blnShuffleData':          argShuffleData,      # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'fltLearningRate':         argLearningRate,     # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'intPrintEvery':           argPrintEvery,       # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'fltGradClip':             argGradClip,         # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)

        'lstTrainingStepLosses':   argTrainingStepLosses,
        'lstValidationStepLosses': argValidationStepLosses,
        
        'dctModelProperties':      argModelProperties   # Important training and model-specific parameters        
    }
    
    if (argDebug): print(dctModelCheckPt)
    
    # Save model to file system with 'write' and 'binary' options
    with open(os.path.join(argModelDir, argModelName), 'wb') as objModelFile:
        torch.save(dctModelCheckPt, objModelFile)


# In[ ]:


def fnLoadLSTMModel(argModelDir, argModelName, argDebug = False):
    print('Loading model: argModelName = {}'.format(argModelName))
    
    with open(os.path.join(argModelDir, argModelName), 'rb') as objModelFile:
        dctModelCheckPt = torch.load(objModelFile, weights_only=False)

    # Extract saved parameters from the model file
    intNumChannels          = dctModelCheckPt['intNumChannels']
    intSeqLen               = dctModelCheckPt['intSeqLen']
    intNumSegments          = dctModelCheckPt['intNumSegments']

    intFeaturesDim          = dctModelCheckPt['intFeaturesDim']
    intHiddenDim            = dctModelCheckPt['intHiddenDim']
    intNumLayers            = dctModelCheckPt['intNumLayers']
    intOutputSize           = dctModelCheckPt['intOutputSize']
    fltDropProb             = dctModelCheckPt['fltDropProb']
    dctStateDict            = dctModelCheckPt['dctStateDict']

    intNumEpochs            = dctModelCheckPt['intNumEpochs']
    intBatchSize            = dctModelCheckPt['intBatchSize']
    blnShuffleIndices       = dctModelCheckPt['blnShuffleIndices']
    blnShuffleData          = dctModelCheckPt['blnShuffleData']
    fltLearningRate         = dctModelCheckPt['fltLearningRate']
    intPrintEvery           = dctModelCheckPt['intPrintEvery']
    fltGradClip             = dctModelCheckPt['fltGradClip']

    lstTrainingStepLosses   = dctModelCheckPt['lstTrainingStepLosses']
    lstValidationStepLosses = dctModelCheckPt['lstValidationStepLosses']
    
    # Return dctModelProperties{} if it exists. Otherwise, return an empty dictionary
    if ('dctModelProperties' in dctModelCheckPt.keys()):
        dctModelProperties  = dctModelCheckPt['dctModelProperties']
        blnHasModelProperties = True
    else:
        dctModelProperties  = {}
        blnHasModelProperties = False
    
    # Reconstruct the LSTM model with the saved parameters
    objModelLSTM = clsLSTM(intFeaturesDim, intHiddenDim, intNumLayers, intOutputSize, argDropProb = fltDropProb)
    objModelLSTM.load_state_dict(dctStateDict)
    
    return (intNumChannels, intSeqLen, intNumSegments, objModelLSTM,
            intNumEpochs, intBatchSize, blnShuffleIndices, blnShuffleData, fltLearningRate, intPrintEvery, fltGradClip,
            lstTrainingStepLosses, lstValidationStepLosses, dctModelProperties)

