local Tanh, parent = torch.class('jzt.Tanh','nn.Module')

function Tanh:__init()
   parent.__init(self)
   self:cuda()
end

function Tanh:updateOutput(input)
   jzt.tanh(input, input)
   self.output = input
   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   jzt.mult_by_tanh_deriv(gradOutput, self.output, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
