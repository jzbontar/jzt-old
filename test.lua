require 'Test'
require 'jzt'
require 'nn'
require 'cunn'

test = {}
function test.Linear()
   A = torch.CudaTensor(5, 3):normal()
   
   module = jzt.Linear(3, 4)
   print(testJacobian(module, A))
   print(testJacobianParameters(module, A))
end


function test.LinearTanh()
   A = torch.CudaTensor(5, 3):normal()

   net = nn.Sequential()
   net:add(jzt.Linear(3, 4))
   net:add(jzt.Tanh())
   net = net:cuda()

   print(testJacobian(net, A))
   print(testJacobianParameters(net, A))
end

function test.StereoJoin()
   A = torch.CudaTensor(4, 3, 5, 5):normal()
   n = jzt.StereoJoin(4)
   print(testJacobian(n, A))
end

function test.SpatialConvolution1()
   A = torch.CudaTensor(4, 3, 5, 5):normal()
   module = jzt.SpatialConvolution1(3, 4)

   print(testJacobian(module, A))
   print(testJacobianParameters(module, A))
end

test = {}
function test.foo()
   A = torch.CudaTensor(16, 1, 20, 30):normal()
   B = torch.CudaTensor(16, 1, 20, 30):normal()
   
   net = nn.Sequential{bprop_min=2,timing=0,debug=0}
   net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
   net:add(nn.SpatialConvolutionRing(1, 16, 5, 5))
   net:add(jzt.Tanh())
   net:add(jzt.StereoJoin(10))
   net:add(jzt.SpatialConvolution1(10, 10))
   net:add(jzt.Tanh())
   net:add(jzt.SpatialConvolution1(10, 1))
   net = net:cuda()

   measure = nn.MSECriterion()
   measure = measure:cuda()   

   print(testNetworkParameters(net, measure, A, B))
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
