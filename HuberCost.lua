local HuberCost, parent = torch.class('jzt.HuberCost', 'nn.Criterion')

function HuberCost:__init(threshold)
   parent.__init(self)
   self:cuda()
   self.threshold = threshold
end

function HuberCost:updateOutput(input, target)
   self.gradInput:resizeAs(input)
   jzt.huber(input, target, self.gradInput, self.threshold)
   self.output = self.gradInput:sum()
   return self.output
end

function HuberCost:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   jzt.huber_deriv(input, target, self.gradInput, self.threshold)
   return self.gradInput
end
