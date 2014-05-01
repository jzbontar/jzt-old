local L1Cost, parent = torch.class('jzt.L1Cost', 'nn.Criterion')

function L1Cost:__init()
   parent.__init(self)
end

function L1Cost:updateOutput(input)
   return input:norm(1)
end

function L1Cost:updateGradInput(input)
   self.gradInput:resizeAs(input)
   return self.gradInput:sign(input)
end
