local SpatialClassNLLCriterion, parent = torch.class('nn.SpatialClassNLLCriterion', 'nn.Criterion')

function SpatialClassNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.output = torch.Tensor(1)
end

function SpatialClassNLLCriterion:updateOutput(input, target)
--   if target:dim() ~= 1 then error('multi-target not implemented') end
   input.nn.SpatialClassNLLCriterion_updateOutput(self, input, target)
   return self.output
end

function SpatialClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   input.nn.SpatialClassNLLCriterion_updateGradInput(self, input, target)
   return self.gradInput
end
