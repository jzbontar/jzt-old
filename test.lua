require 'jzt'
require 'Test'

test = {}
function test.HuberCost()
   A = torch.CudaTensor(5, 3):normal()
   B = torch.CudaTensor(5, 3):normal()

   print(testCriterion(jzt.HuberCost(1), A, B))
end

function test.KLDivergence()
   A = torch.CudaTensor(5, 3):normal():log()
   B = torch.CudaTensor(5, 3):normal():log()

   print(testCriterion(jzt.KLDivergence(), A, B))
end

-- function test.ClassNLLCriterion()
--    A = torch.CudaTensor(5, 3):normal()
--    B = torch.eye(3):index(1, torch.LongTensor{1,2,3,1,2})
-- 
--    print(testCriterion(jzt.ClassNLLCriterion(), A, B))
-- end

function test.Linear()
   A = torch.CudaTensor(5, 3):normal()
   
   module = jzt.Linear(3, 4)
   print(testJacobian(module, A))
   print(testJacobianParameters(module, A))
end

function test.SpatialConvolution1()
   A = torch.CudaTensor(3, 4, 5, 5):normal()
   module = jzt.SpatialConvolution1(4, 2)

   print(testJacobian(module, A))
   print(testJacobianParameters(module, A))
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
