require 'jzt'

local SpatialBias, parent = torch.class('jzt.SpatialBias', 'nn.Module')

function SpatialBias:__init(dim1, dim2, dim3)
   parent.__init(self)
   self.bias = torch.CudaTensor(1, dim1, dim2, dim3):zero()
   self.gradBias = torch.CudaTensor(dim1 * dim2 * dim3)
   self:cuda()
end

function SpatialBias:updateOutput(input)
   self.output = input:add(1, self.bias:expandAs(input))
   return self.output
end

function SpatialBias:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function SpatialBias:accGradParameters(input, gradOutput)
   -- self.gradBias:sum(gradOutput, 1)
   dim1 = input:size(1)
   dim2 = input:size(2)
   dim3 = input:size(3)
   dim4 = input:size(4)

   jzt.sum(gradOutput:resize(dim1, dim2 * dim3 * dim4), self.gradBias, 1)
end
