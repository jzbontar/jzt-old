local MSECost, parent = torch.class('jzt.MSECost', 'nn.Criterion')

function MSECost:__init()
   parent.__init(self)
   self:cuda()
end

function MSECost:updateOutput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:add(input, -1, target)
--   self.output = 0.5 * self.gradInput:dot(self.gradInput)
--   return self.output
end

function MSECost:updateGradInput(input, target)
   return self.gradInput
end
