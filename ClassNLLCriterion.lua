local ClassNLLCriterion, parent = torch.class('jzt.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.tmp = torch.CudaTensor()
end

function ClassNLLCriterion:updateOutput(input, target)
   if input:dim() == 1 then
      self.output = -input[target]
   elseif input:dim() == 2 then
      self.tmp:resizeAs(target)
      jzt.get_cols(input, target, self.tmp)
      local output = -self.tmp:sum()

      if self.sizeAverage then
         output = output / target:size(1)
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function ClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

  if input:dim() == 1 then
      self.gradInput[target] = -1
   else
      local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      jzt.set_cols(self.gradInput, target, z)
   end

   return self.gradInput
end
