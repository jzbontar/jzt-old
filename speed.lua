require 'jzt'
require 'prof-torch'

test = {}
function test.StereoJoin()
   A = torch.CudaTensor(16, 16, 250, 1224):normal()
   n = jzt.StereoJoin(64)

   n:updateOutput(A)
   n:updateGradInput(A, n.output)
   cutorch.synchronize()
   
   for i = 1,100 do
      prof.tic('f')
      n:updateOutput(A)
      cutorch.synchronize()
      prof.toc('f')

      prof.tic('b')
      n:updateGradInput(A, n.output)
      cutorch.synchronize()
      prof.toc('b')
   end

   prof.dump()
   
   return
end

test = {}
function test.SpatialConvolution1()
   A = torch.CudaTensor(16, 16, 250, 1224):normal()
   net = jzt.SpatialConvolution1(16, 16)

   net:forward(A)
   net:backward(A, net.output)
   cutorch.synchronize()

   prof.tic()
   for i = 1,20 do
      net:forward(A)
      net:backward(A, net.output)
   end
   cutorch.synchronize()
   print(prof.toc())
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
