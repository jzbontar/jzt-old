require 'jzt'

local Linear, parent = torch.class('jzt.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize, doBias)
   parent.__init(self)
   self:cuda()

   self.doBias = doBias == nil and true or doBias

   self.weight = torch.CudaTensor(outputSize, inputSize)
   self.gradWeight = torch.CudaTensor(outputSize, inputSize)
   if self.doBias then
      self.bias = torch.CudaTensor(outputSize)
      self.gradBias = torch.CudaTensor(outputSize)
   end

   self:reset()
end

function Linear:reset()
   stdv = 1 / math.sqrt(self.weight:size(2))
   self.weight:uniform(-stdv, stdv)
   if self.doBias then
      self.bias:uniform(-stdv, stdv)
   end
end

function Linear:updateOutput(input)
   local nframe = input:size(1)
   local nunit = self.weight:size(1)

   self.output:resize(nframe, nunit)
   self.output:addmm(0, 1, input, self.weight:t())
   if self.doBias then
      jzt.add_mat_vect(self.output, self.bias, self.output, 1)
   end
   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:addmm(0, 1, gradOutput, self.weight)
   return self.gradInput
end

function Linear:accGradParameters(input, gradOutput)
   self.gradWeight:addmm(0, 1, gradOutput:t(), input)
   if self.doBias then
      jzt.sum(gradOutput, self.gradBias, 1)
   end
end
