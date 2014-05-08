local SpatialClassNLLCriterion, parent = torch.class('jzt.SpatialClassNLLCriterion', 'nn.Criterion')

function SpatialClassNLLCriterion:__init()
   parent.__init(self)
   self:cuda()
   self.sizeAverage = true
   self.tmp = torch.CudaTensor()
end

function SpatialClassNLLCriterion:updateOutput(input, target)
   self.tmp:resizeAs(target)
   jzt.get_spatial(input, target, self.tmp)
   self.output = -self.tmp:sum()

   if self.sizeAverage then
      self.tmp:ne(target, 0)
      self.output = self.output / self.tmp:sum()
   end
   return self.output
end

function SpatialClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   local z = -1
   if self.sizeAverage then
      self.tmp:ne(target, 0)
      z = z / self.tmp:sum()
   end
   jzt.set_spatial(self.gradInput, target, z)

   return self.gradInput
end
