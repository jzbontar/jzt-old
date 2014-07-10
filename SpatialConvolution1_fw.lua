require 'jzt'

local SpatialConvolution1_fw, parent = torch.class('jzt.SpatialConvolution1_fw', 'nn.Module')

function SpatialConvolution1_fw:__init(inputSize, outputSize)
   parent.__init(self)
   self:cuda()

   self.weight = torch.CudaTensor(outputSize, inputSize)
   self.bias = torch.CudaTensor(1, outputSize, 1, 1)
end

function SpatialConvolution1_fw:updateOutput(input)
   assert(input:size(1) == 1)
   local fm_in = input:size(2)
   local h = input:size(3)
   local w = input:size(4)
   local fm_out = self.weight:size(1)

   self.output:resize(fm_out, h * w)
   self.output:addmm(0, 1, self.weight, input:resize(fm_in, h * w))
   self.output:resize(1, fm_out, h, w)
   self.output:add(self.bias:expandAs(self.output))

   return self.output
end
