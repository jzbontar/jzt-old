local Margin2Loss, parent = torch.class('jzt.Margin2Loss', 'nn.Criterion')

function Margin2Loss:__init(margin)
   parent.__init(self)
   self:cuda()
   self.margin = margin
   self.tmp = torch.CudaTensor()
end

function Margin2Loss:updateOutput(input, target)
   self.gradInput:resizeAs(input)
   self.tmp:resizeAs(target)

   local ntargets = self.tmp:ne(target, 0):sum()

   self.gradInput:zero()
   self.tmp:zero()
   jzt.SpatialMargin2_costGrad(input, target, self.gradInput, self.tmp, self.margin)
   self.output = self.tmp:sum() / ntargets
   self.gradInput:div(ntargets)
   return self.output
end

function Margin2Loss:updateGradInput(input, target)
   return self.gradInput
end
