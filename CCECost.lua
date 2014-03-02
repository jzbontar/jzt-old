-- Categorical crossentropy cost
local CCECost, parent = torch.class('jzt.CCECost', 'nn.Criterion')

function CCECost:__init()
   parent.__init(self)
   self:cuda()

   self.softmax_out = torch.CudaTensor()
   self.tmp_softmax = torch.CudaTensor()
   self.tmp_cce = torch.CudaTensor()
end

function CCECost:updateOutput(input, target)
   self.softmax_out:resizeAs(input)
   self.tmp_softmax:resize(input:size(2))
   self.tmp_cce:resizeAs(input)

   jzt.softmax(input, self.softmax_out, self.tmp_softmax)
   self.output = jzt.cce(self.softmax_out, target, self.tmp_cce)

   return self.output
end

function CCECost:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   jzt.sub(self.softmax_out, target, self.gradInput)
   return self.gradInput
end
