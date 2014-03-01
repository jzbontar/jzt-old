require 'libjzt'
require 'nn'
require 'torch'
require 'cutorch'

include('Linear.lua')

include('ClassNLLCriterion.lua')
include('L1Cost.lua')
include('MSECost.lua')
include('HuberCost.lua')
