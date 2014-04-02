require 'jzt'

local StereoJoin, parent = torch.class('jzt.StereoJoin', 'nn.Module')

function StereoJoin:__init(n)
   parent.__init(self)
   self.n = n
   self:cuda()
   self.gradInputLeft = torch.CudaTensor()
   self.gradInputRight = torch.CudaTensor()
end

function StereoJoin:updateOutput(input)
   local batch_size = input:size(1) / 2
   self.output:resize(batch_size, self.n, input:size(3), input:size(4))

   local left = input:narrow(1, 1, batch_size)
   local right = input:narrow(1, 1 + batch_size, batch_size)

   jzt.stereoJoin_updateOutput(left, right, self.output)
   return self.output
end

function StereoJoin:updateGradInput(left, right, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:addmm(0, 1, gradOutput, self.weight)
   return self.gradInput
end
