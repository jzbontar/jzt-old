local Dropout, Parent = torch.class('jzt.Dropout', 'nn.Module')
 
function Dropout:__init(percentage)
   Parent.__init(self)
   self.p = percentage
   self.noise = torch.CudaTensor()
end
 
function Dropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)

   -- stage is a global variable
   if stage == 'train' then
      self.noise:resizeAs(input)
      self.noise:bernoulli(self.p)
      self.output:cmul(self.noise)
   elseif stage == 'test' then
      self.output:mul(self.p)
   else
      assert(false)
   end
   return self.output
end
 
function Dropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   return self.gradInput
end
