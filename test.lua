require 'jzt'
require 'Test'

test = {}
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

test = {}
function test.LinearTanh()
   A = torch.CudaTensor(5, 3):normal()

   module1 = jzt.Linear(3, 4)
   module2 = jzt.Tanh(module1)
   module = jzt.JoinModules(module1, module2)

   print(testJacobian(module, A))
   print(testJacobianParameters(module, A))
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
