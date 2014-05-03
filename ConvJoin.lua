local ConvJoin, parent = torch.class('jzt.ConvJoin','nn.Module')

function ConvJoin:__init(split)
   parent.__init(self)
   self.split = split
   self:cuda()
end

function ConvJoin:updateOutput(input)
   self.output:resize(self.split.nimg, input:size(2), 
      self.split.height - 2 * self.split.overlap, 
      self.split.width - 2 * self.split.overlap)
   jzt.ConvJoin_updateOutput(input, self.output, self.split.nrow, self.split.ncol)
   return self.output
end
