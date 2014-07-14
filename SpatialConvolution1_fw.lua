require 'jzt'

local SpatialConvolution1_fw, parent = torch.class('jzt.SpatialConvolution1_fw', 'nn.Module')

function SpatialConvolution1_fw:__init(inputSize, outputSize)
   parent.__init(self)
   self:cuda()

   self.tmp_in = torch.CudaTensor()
   self.tmp_out = torch.CudaTensor()
   self.weight = torch.CudaTensor(outputSize, inputSize)
   self.bias = torch.CudaTensor(1, outputSize, 1, 1)
end

function SpatialConvolution1_fw:updateOutput(input)
   local num_ex = input:size(1)
   local fm_in = input:size(2)
   local h = input:size(3)
   local w = input:size(4)
   local fm_out = self.weight:size(1)

   self.tmp_in:resize(fm_in, h * w)
   self.tmp_out:resize(fm_out, h * w)
   self.output:resize(num_ex, fm_out, h, w)
   for i = 1,num_ex do
      self.tmp_in:copy(input[i])
      self.tmp_out:addmm(0, 1, self.weight, self.tmp_in)
      self.output[i]:copy(self.tmp_out)
   end

   self.output:add(self.bias:expandAs(self.output))
   return self.output
end
