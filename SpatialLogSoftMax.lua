local SpatialLogSoftMax,parent = torch.class('jzt.SpatialLogSoftMax', 'nn.Module')

function SpatialLogSoftMax:__init(constant)
   parent.__init(self)
   self.constant_present = false
   self.constant = constant -- a constant added to the exponential sum
   if self.constant then
      print('Using exp('..self.constant
	    ..') as additional denomimator constant in SpatialLogSoftMax')
      self.constant_present = true
   end
end

function SpatialLogSoftMax:updateOutput(input)
   jzt.SpatialLogSoftMax_updateOutput(self, input)
   self.output = input
   return self.output
end

function SpatialLogSoftMax:updateGradInput(input, gradOutput)
   jzt.SpatialLogSoftMax_updateGradInput(self, input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
