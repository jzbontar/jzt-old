local Tanh, parent = torch.class('jzt.Tanh','nn.Module')

function Tanh:__init(prev_module)
   parent.__init(self)
   self:cuda()

   self.output = prev_module.output
   self.gradInput = prev_module.gradInput
end

function Tanh:updateOutput(input)
   jzt.tanh(self.output, self.output)
end

function Tanh:updateGradInput(input, gradOutput)
   jzt.mult_by_tanh_deriv(gradOutput, self.output, gradOutput)
end
