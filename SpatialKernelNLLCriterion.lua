local SpatialKernelNLLCriterion, parent = torch.class('jzt.SpatialKernelNLLCriterion', 'nn.Criterion')

function SpatialKernelNLLCriterion:__init(kernel)
   parent.__init(self)
   self.tmp = torch.CudaTensor()
   assert(kernel:nElement() % 2 == 1)
   self.kernel = kernel
end

function SpatialKernelNLLCriterion:updateOutput(input, target)
   self.tmp:resizeAs(target)
   jzt.get_spatial_kernel(input, target, self.kernel, self.tmp)
   self.output = -self.tmp:sum()
   self.tmp:ne(target, 0)
   self.cnt = self.tmp:sum()
   self.output = self.output / self.cnt
   return self.output
end

function SpatialKernelNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   jzt.set_spatial_kernel(self.gradInput, target, self.kernel)
   self.tmp:ne(target, 0)
   self.gradInput:div(-self.cnt)
   return self.gradInput
end
