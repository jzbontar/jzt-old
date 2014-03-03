-- Assume input and target are logs
local KLDivergence, parent = torch.class('jzt.KLDivergence', 'nn.Criterion')

function KLDivergence:__init()
   parent.__init(self)
   self:cuda()
   self.diff = torch.CudaTensor()
   self.exp = torch.CudaTensor()
end

function KLDivergence:updateOutput(input, target)
   self.diff:resizeAs(input)
   self.exp:resizeAs(input)

   self.diff:add(target, -1, input)
   jzt.exp(target, self.exp)
   self.output = self.diff:dot(self.exp)
   return self.output
end

function KLDivergence:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   jzt.smul(self.exp, self.gradInput, -1)
   return self.gradInput
end
