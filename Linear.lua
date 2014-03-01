require 'jzt'

local Linear, parent = torch.class('jzt.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize)
   parent.__init(self)
   self:cuda()

   self.weight = torch.CudaTensor(outputSize, inputSize)
   self.bias = torch.CudaTensor(outputSize)
   self.gradWeight = torch.CudaTensor(outputSize, inputSize)
   self.gradBias = torch.CudaTensor(outputSize)

   self:reset()
end

function Linear:reset()
   stdv = 1 / math.sqrt(self.weight:size(2))
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function Linear:updateOutput(input)
   local nframe = input:size(1)
   local nunit = self.bias:size(1)

   self.output:resize(nframe, nunit)
   self.output:addmm(0, 1, input, self.weight:t())
   jzt.add_mat_vect(self.output, self.bias, self.output, 1)
   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:addmm(0, 1, gradOutput, self.weight)
   return self.gradInput
end

function Linear:accGradParameters(input, gradOutput)
   self.gradWeight:addmm(0, 1, gradOutput:t(), input)
   jzt.sum(gradOutput, self.gradBias, 1)
end
