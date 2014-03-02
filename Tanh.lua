local Tanh, parent = torch.class('ct.Tanh','nn.Module')

function Tanh:__init()
   parent.__init(self)
   self.output = nil
   self.gradInput = nil
end

function Tanh:updateOutput(input)
   self.output = self.output or ct.emptyAs(input)
   ct.tanh(input, self.output)
   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or ct.emptyAs(input)
   ct.mult_by_tanh_deriv(gradOutput, self.output, self.gradInput)
   return self.gradInput
end
