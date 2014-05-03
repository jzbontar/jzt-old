local ConvJoin, parent = torch.class('jzt.ConvJoin','nn.Module')

function ConvJoin:__init(height, width)
   parent.__init(self)
   self.height = height
   self.width = width
   self:cuda()
end

function ConvJoin:updateOutput(input)
   self.output:resize(1, input:size(2), self.height, self.width)
   jzt.ConvJoin_updateOutput(input, self.output)
   return self.output
end
