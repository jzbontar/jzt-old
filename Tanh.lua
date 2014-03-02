local Tanh, parent = torch.class('jzt.Tanh','nn.Module')

function Tanh:__init()
   parent.__init(self)
   self:cuda()
end

function Tanh:updateOutput(input)
   self.output:resizeAs(input)
   jzt.tanh(input, self.output)
   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   jzt.mult_by_tanh_deriv(gradOutput, self.output, self.gradInput)
   return self.gradInput
end
