local ClassNLLCriterion, parent = torch.class('jzt.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init()
   parent.__init(self)
   self:cuda()
   self.sizeAverage = true
   self.tmp = torch.CudaTensor()
end

function ClassNLLCriterion:updateOutput(input, target)
   self.tmp:resizeAs(target)
   jzt.get_cols(input, target, self.tmp)
   self.output = -self.tmp:sum()

   if self.sizeAverage then
      self.output = self.output / target:size(1)
   end
   return self.output
end

function ClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   local z = -1
   if self.sizeAverage then
      z = z / target:size(1)
   end
   jzt.set_cols(self.gradInput, target, z)

   return self.gradInput
end
