local Margin1Loss, parent = torch.class('jzt.Margin1Loss', 'nn.Criterion')

function Margin1Loss:__init(margin)
   parent.__init(self)
   self:cuda()
   self.margin = margin
   self.tmp = torch.CudaTensor()
end

function Margin1Loss:updateOutput(input, target)
   self.gradInput:resizeAs(input)
   self.tmp:resizeAs(target)
   self.gradInput:zero()
   self.tmp:zero()
   jzt.SpatialMargin1_costGrad(input, target, self.gradInput, self.tmp, self.margin)
   self.output = self.tmp:sum()
   return self.output
end

function Margin1Loss:updateGradInput(input, target)
   return self.gradInput
end
