require 'jzt'

local SpatialConvolution1_fw, parent = torch.class('jzt.SpatialConvolution1_fw', 'nn.Module')

function SpatialConvolution1_fw:__init(inputSize, outputSize)
   parent.__init(self)
   self:cuda()

   self.weight = torch.CudaTensor(outputSize, inputSize)
   self.bias = torch.CudaTensor(1, outputSize, 1, 1)
end

function SpatialConvolution1_fw:updateOutput(input)
   local num_ex = input:size(1)
   local fm_in = input:size(2)
   local h = input:size(3)
   local w = input:size(4)
   local fm_out = self.weight:size(1)

--   if fm_in == fm_out then
--      self.output = input
--   end

   input:resize(num_ex, fm_in, h * w)
   self.output:resize(num_ex, fm_out, h * w)
   for i = 1,num_ex do
      self.output[i]:addmm(0, 1, self.weight, input[i])
   end
   input:resize(num_ex, fm_in, h, w)
   self.output:resize(num_ex, fm_out, h, w)

   self.output:add(self.bias:expandAs(self.output))

   input:storage():resize(0)
--   input:resize(0)
--   collectgarbage()

   return self.output
end
