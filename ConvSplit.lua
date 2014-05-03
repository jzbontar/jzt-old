local ConvSplit, parent = torch.class('jzt.ConvSplit','nn.Module')

function ConvSplit:__init(win_size, overlap)
   parent.__init(self)
   self.win_size = win_size
   self.overlap = overlap
   self:cuda()
end

function ConvSplit:updateOutput(input)
   self.nimg = input:size(1)
   self.height = input:size(3)
   self.width = input:size(4)
   self.nrow = math.ceil(self.height / (self.win_size - 2 * self.overlap))
   self.ncol = math.ceil(self.width / (self.win_size - 2 * self.overlap))

   local bs = self.nimg * self.nrow * self.ncol
   -- make bs multiple of 4 (SpatialConvolutionFFT fails otherwise)
   bs = math.ceil(bs / 4) * 4 
   assert(bs % 4 == 0)
   self.output:resize(bs, input:size(2), self.win_size, self.win_size)
   jzt.ConvSplit_updateOutput(input, self.output, self.win_size, self.overlap, self.nrow, self.ncol)
   return self.output
end
