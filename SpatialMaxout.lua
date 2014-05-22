local SpatialMaxout, parent = torch.class('jzt.SpatialMaxout','nn.Module')

function SpatialMaxout:__init(poolsize)
   parent.__init(self)
   self:cuda()
   self.poolsize = poolsize
end

function SpatialMaxout:updateOutput(input)
   assert(input:size(2) % self.poolsize == 0)
   self.output:resize(input:size(1), input:size(2) / self.poolsize, 
      input:size(3), input:size(4))

   jzt.SpatialMaxout_costGrad(input, self.output, self.gradInput, 
      self.gradInput, self.poolsize, 0)
   return self.output
end

function SpatialMaxout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   
   jzt.SpatialMaxout_costGrad(input, self.output, self.gradInput, 
      gradOutput, self.poolsize, 1)
   return self.gradInput
end
