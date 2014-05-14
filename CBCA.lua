local CBCA, parent = torch.class('jzt.CBCA','nn.Module')

function CBCA:__init(img, tau, L1)
   parent.__init(self)

   self.tau = tau
   self.L1 = L1
   self.img = img
   self:cuda()
end

function CBCA:updateOutput(input)
   self.output:resizeAs(input)
   jzt.cbca_costGrad(self.img, input, self.output, self.gradInput, self.gradInput, self.tau, self.L1, 0)
   return self.output
end

function CBCA:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   jzt.cbca_costGrad(self.img, input, self.output, self.gradInput, gradOutput, self.tau, self.L1, 1)
   return self.gradOutput
end
