require 'jzt'

local StereoJoin, parent = torch.class('jzt.StereoJoin', 'nn.Module')

function StereoJoin:__init(n, dist)
   parent.__init(self)
   self.dist = dist
   self.n = n
   self:cuda()
   self.gradInputLeft = torch.CudaTensor()
   self.gradInputRight = torch.CudaTensor()
end

function StereoJoin:updateOutput(input)
   local batch_size = input:size(1) / 2
   local left = input:narrow(1, 1, batch_size)
   local right = input:narrow(1, 1 + batch_size, batch_size)
   self.output:resize(batch_size, self.n, input:size(3), input:size(4))

   jzt.stereoJoin_updateOutput(left, right, self.output, self.dist)
   return self.output
end

function StereoJoin:updateGradInput(input, gradOutput)
   local batch_size = input:size(1) / 2
   local left = input:narrow(1, 1, batch_size)
   local right = input:narrow(1, 1 + batch_size, batch_size)

   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   local leftGrad = self.gradInput:narrow(1, 1, batch_size)
   local rightGrad = self.gradInput:narrow(1, 1 + batch_size, batch_size)

   jzt.stereoJoin_updateGradInput(left, right, gradOutput, leftGrad, rightGrad, self.dist)
   return self.gradInput
end
