local Sqrt, parent = torch.class('jzt.Sqrt','nn.Module')

function Sqrt:__init()
   parent.__init(self)
   self.output_clone = torch.CudaTensor()
end

function Sqrt:updateOutput(input)
   self.output = input:sqrt()
   self.output_clone:resizeAs(self.output):copy(self.output)
   return self.output
end

function Sqrt:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:cdiv(self.output_clone):div(2)
   return self.gradInput
end
