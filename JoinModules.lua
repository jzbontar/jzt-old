require 'jzt'

local JoinModules, parent = torch.class('jzt.JoinModules', 'nn.Module')

function JoinModules:__init(module1, module2)
   parent.__init(self)
   self:cuda()

   self.module1 = module1
   self.module2 = module2

   self.weight = module1.weight
   self.bias = module1.bias
   self.gradWeight = module1.gradWeight
   self.gradBias = module1.gradBias

   self.output = module1.output
   self.gradInput = module1.gradInput
end

function JoinModules:updateOutput(input)
   self.module1:updateOutput(input)
   self.module2:updateOutput(input)
   return self.output
end

function JoinModules:updateGradInput(input, gradOutput)
   self.module2:updateGradInput(input, gradOutput)
   self.module1:updateGradInput(input, gradOutput)
   return self.gradInput
end

function JoinModules:accGradParameters(input, gradOutput)
   self.module1:accGradParameters(input, gradOutput)
end
