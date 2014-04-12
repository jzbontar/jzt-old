local Relu, parent = torch.class('jzt.Relu','nn.Module')

function Relu:__init()
   parent.__init(self)
   self:cuda()
end

function Relu:updateOutput(input)
   jzt.relu(input, input)
   self.output = input
   return self.output
end

function Relu:updateGradInput(input, gradOutput)
   jzt.mult_by_relu_deriv(gradOutput, self.output, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
