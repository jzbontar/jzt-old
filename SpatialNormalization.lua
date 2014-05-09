require 'jzt'

local SpatialNormalization, parent = torch.class('jzt.SpatialNormalization', 'nn.Module')

function SpatialNormalization:__init(nInputPlane)
   parent.__init(self)
   self:cuda()
   self.div = nn.CDivTable():cuda()
   self.net = nn.Sequential()
   self.net:add(nn.Square())
   self.net:add(nn.Sum(2))
   self.net:add(nn.Sqrt())
   self.net:add(nn.Replicate(nInputPlane))
   self.net:add(nn.Transpose({1, 2}))
   self.net:add(nn.Threshold(1e-5, 1e-5))
   self.net:cuda()
end

function SpatialNormalization:updateOutput(input)
   self.output = self.div:updateOutput({input, self.net:updateOutput(input)})
   return self.output
end

function SpatialNormalization:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   local g = self.div:updateGradInput({input, self.net.output}, gradOutput)
   self.gradInput:add(g[1])
   self.gradInput:add(self.net:updateGradInput(input, g[2]))
   return self.gradInput
end
