local L2Pooling,parent = torch.class('jzt.L2Pooling', 'nn.Module')

function L2Pooling:__init(ksize, stride)
   parent.__init(self)
   self.ksize = ksize
   self.stride = stride
   self:cuda()
end

function L2Pooling:updateOutput(input)
   pooled_height = math.floor((input:size(3) - self.ksize) / self.stride) + 1
   pooled_width = math.floor((input:size(4) - self.ksize) / self.stride) + 1
   self.output:resize(input:size(1), input:size(2), pooled_height, pooled_width)
   jzt.L2Pooling_updateOutput(input, self.output, self.ksize, self.stride)
   return self.output
end

function L2Pooling:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   jzt.L2Pooling_updateGradInput(input, self.output, self.gradInput, gradOutput, self.ksize, self.stride)
   return self.gradInput
end
