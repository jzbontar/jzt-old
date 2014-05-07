local Margin1Loss, parent = torch.class('jzt.Margin1Loss', 'nn.Criterion')

function Margin1Loss:__init()
   parent.__init(self)
   self:cuda()
   self.actual = torch.CudaTensor()
   self.argmax = torch.CudaTensor()
   self.offending = torch.CudaTensor()
end

function Margin1Loss:updateOutput(input, target)
   assert(input:nDimension() == 4)
   assert(input:size(1) == 1)
   assert(target:size(1) == 1)
   assert(target:size(2) == 1)

   -- I will destroy the input -- make a copy
   input = input:clone()

   -- actual
   self.actual:resizeAs(target)
   jzt.get_spatial(input, target, self.actual)
   jzt.set_spatial(input, target, -2e38)

   print(self.actual)

   -- most offending
   self.argmax:resizeAs(target)
   self.offending:resizeAs(target)
   jzt.spatial_argmax(input, self.argmax)
   jzt.get_spatial(input, self.argmax, self.offending)

   self.offending:add(-1, self.actual):add(1)
   jzt.relu(self.offending, self.offending)

   self.output = self.offending:sum()
   return self.output
end

function Margin1Loss:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   local z = -1
   if self.sizeAverage then
      z = z / target:size(1)
   end
   jzt.set_cols(self.gradInput, target, z)

   return self.gradInput
end
