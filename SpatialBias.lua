require 'jzt'

local SpatialBias, parent = torch.class('jzt.SpatialBias', 'nn.Module')

function SpatialBias:__init()
   parent.__init(self)
   self.bias = torch.CudaTensor()
   self.gradBias = torch.CudaTensor()
   self:cuda()
end

function SpatialBias:updateOutput(input)
   local nElement = self.bias:nElement()
   self.bias:resizeAs(input)
   self.gradBias:resizeAs(input)
   if nElement ~= self.bias:nElement() then
      self.bias:zero()
   end
   
   self.output = input
   input:add(1, self.bias)
   return self.output
end

function SpatialBias:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function SpatialBias:accGradParameters(input, gradOutput)
   self.gradBias:copy(gradOutput)
   return self.gradBias
end
