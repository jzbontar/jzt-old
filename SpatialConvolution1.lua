require 'jzt'

local SpatialConvolution1, parent = torch.class('jzt.SpatialConvolution1', 'nn.Module')

function SpatialConvolution1:__init(inputSize, outputSize)
   parent.__init(self)
   self:cuda()

   self.weight = torch.CudaTensor(outputSize, inputSize)
   self.gradWeight = torch.CudaTensor(outputSize, inputSize)
   self.bias = torch.CudaTensor(outputSize)
   self.gradBias = torch.CudaTensor(outputSize)

   self.gradBiasTmp = torch.CudaTensor()

   self:reset()
end

function SpatialConvolution1:reset()
   stdv = 1 / math.sqrt(self.weight:size(2))
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function SpatialConvolution1:updateOutput(input)
   self.output:resize(input:size(1), self.weight:size(1), input:size(3), input:size(4))
   jzt.sc1_updateOutput(input, self.weight, self.output)
   jzt.add_bias4(self.output, self.bias)
   return self.output
end

function SpatialConvolution1:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   jzt.sc1_updateOutput(gradOutput, self.weight:t():clone(), self.gradInput)
   return self.gradInput
end

function SpatialConvolution1:accGradParameters(input, gradOutput)
   local img_size = input:size(3) * input:size(4)

   jzt.sc1_accGradParameters(input, gradOutput, self.gradWeight) 

   gradOutput:resize(gradOutput:size(1), gradOutput:size(2), img_size)
   self.gradBiasTmp:resize(gradOutput:size(1), gradOutput:size(2))
   self.gradBiasTmp:sum(gradOutput, 3)
   self.gradBias:sum(self.gradBiasTmp, 1)
   gradOutput:resizeAs(self.output)
end
