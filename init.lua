require 'torch'
require 'libjzt'
require 'nn'
require 'cutorch'

include('util.lua')

include('Linear.lua')
include('Relu.lua')
include('SpatialConvolution1.lua')
include('SpatialLogSoftMax.lua')
include('StereoJoin.lua')
include('SpatialBias.lua')
include('Tanh.lua')
include('L2Pooling.lua')
include('ConvSplit.lua')
include('ConvJoin.lua')
include('SpatialNormalization.lua')
include('Mul.lua')

include('ClassNLLCriterion.lua')
include('L1Cost.lua')
include('MSECost.lua')
include('HuberCost.lua')
include('KLDivergence.lua')
include('Margin1Loss.lua')
