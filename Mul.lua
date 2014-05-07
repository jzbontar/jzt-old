local Mul, parent = torch.class('jzt.Mul','nn.Module')

function Mul:__init(n)
   parent.__init(self)
   self.n = n
end

function Mul:updateOutput(input)
   self.output = input:mul(self.n)
   return self.output
end

function Mul:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:mul(self.n)
   return self.gradInput
end
