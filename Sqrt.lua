local Sqrt, parent = torch.class('jzt.Sqrt','nn.Module')

function Sqrt:__init()
   parent.__init(self)
end

function Sqrt:updateOutput(input)
   self.output = input:sqrt()
   return self.output
end

function Sqrt:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:cdiv(self.output):div(2)
   return self.gradInput
end
