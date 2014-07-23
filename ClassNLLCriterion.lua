local ClassNLLCriterion, parent = torch.class('jzt.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init()
   parent.__init(self)
   self.tmp = torch.CudaTensor()
end

function ClassNLLCriterion:updateOutput(input, target)
   self.tmp:resizeAs(target)
   jzt.get_cols(input, target, self.tmp)
   self.output = -self.tmp:sum()
   self.size = self.tmp:ne(target, 0):sum()
   self.output = self.output / self.size
   return self.output
end

function ClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   jzt.set_cols(self.gradInput, target, -1 / self.size)
   return self.gradInput
end
