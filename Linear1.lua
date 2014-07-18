require 'jzt'

local Linear1, parent = torch.class('jzt.Linear1', 'nn.Module')

function Linear1:__init(inputSize)
   parent.__init(self)
   self:cuda()

   self.weight = torch.CudaTensor(inputSize)
   self.gradWeight = torch.CudaTensor(inputSize)
   self.bias = torch.CudaTensor(1)
   self.gradBias = torch.CudaTensor(1)

   self:reset()
end

function Linear1:reset()
   stdv = 1 / math.sqrt(self.weight:nElement())
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function Linear1:updateOutput(input)
   local nframe = input:size(1)

   self.output:resize(nframe)
   self.output:addmv(0, 1, input, self.weight)
   self.output:add(self.bias[1])
   return self.output
end

function Linear1:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:zero():addr(gradOutput, self.weight)
   return self.gradInput
end

function Linear1:accGradParameters(input, gradOutput)
   self.gradWeight:addmv(0, 1, input:t(), gradOutput)
   self.gradBias[1] = gradOutput:sum()
end
