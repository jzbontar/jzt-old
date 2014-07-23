require 'jzt'

local SpatialConvolution1, parent = torch.class('jzt.SpatialConvolution1', 'nn.Module')

function SpatialConvolution1:__init(fm_in, fm_out)
   parent.__init(self)
   self:cuda()

   self.weight = torch.CudaTensor(fm_out, fm_in)
   self.gradWeight = torch.CudaTensor(fm_out, fm_in)
   self.bias = torch.CudaTensor(1, fm_out, 1, 1)
   self.gradBias = torch.CudaTensor(1, fm_out, 1, 1)
   self.fm_in = fm_in
   self.fm_out = fm_out

   self:reset()
end

function SpatialConvolution1:reset()
   stdv = 1 / math.sqrt(self.weight:size(2))
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function SpatialConvolution1:updateOutput(input)
   assert(input:size(1) == 1)
   local h = input:size(3)
   local w = input:size(4)

   input:resize(self.fm_in, h * w)
   self.output:resize(self.fm_out, h * w)
   self.output:addmm(0, 1, self.weight, input)
   self.output:resize(1, self.fm_out, h, w)
   input:resize(1, self.fm_in, h, w)

   self.output:add(self.bias:expandAs(self.output))
   return self.output
end

function SpatialConvolution1:updateGradInput(input, gradOutput)
   assert(gradOutput:size(1) == 1)
   local h = input:size(3)
   local w = input:size(4)

   self.gradInput:resize(self.fm_in, h * w)
   gradOutput:resize(self.fm_out, h * w)
   self.gradInput:addmm(0, 1, self.weight:t(), gradOutput)
   self.gradInput:resize(1, self.fm_in, h, w)
   gradOutput:resize(1, self.fm_out, h, w)
   return self.gradInput
end

function SpatialConvolution1:accGradParameters(input, gradOutput)
   local h = input:size(3)
   local w = input:size(4)

   input:resize(self.fm_in, h * w)
   gradOutput:resize(self.fm_out, h * w)
   self.gradWeight:addmm(0, 1, gradOutput, input:t())
   self.gradBias:sum(gradOutput, 2)
   input:resize(1, self.fm_in, h, w)
   gradOutput:resize(1, self.fm_out, h, w)
end
