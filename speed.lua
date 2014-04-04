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

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
