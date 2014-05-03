local ConvSplit, parent = torch.class('jzt.ConvSplit','nn.Module')

function ConvSplit:__init(win_size, overlap)
   parent.__init(self)
   self.win_size = win_size
   self.overlap = overlap
   self:cuda()
end

function ConvSplit:updateOutput(input)
   assert(input:nDimension() == 4)
   assert(input:size(2) == 1)

   local nrow = math.ceil(input:size(3) / (self.win_size - 2 * self.overlap))
   local ncol = math.ceil(input:size(4) / (self.win_size - 2 * self.overlap))

   self.output:resize(nrow * ncol, 1, self.win_size, self.win_size)
   jzt.ConvSplit_updateOutput(input, self.output, self.win_size, self.overlap, nrow, ncol)
   return self.output
end
