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

function test.SpatialConvolution1()
   A = torch.CudaTensor(4, 3, 5, 5):normal()
   module = jzt.SpatialConvolution1(3, 4)

   print(testJacobian(module, A))
   print(testJacobianParameters(module, A))
end

function test.SpatialLogSoftMax()
   A = torch.CudaTensor(4, 3, 4, 4):normal()
   
   net = nn.Sequential()
   net:add(nn.SpatialConvolution(3, 4, 2, 2))
   net:add(jzt.SpatialLogSoftMax())
   net = net:cuda()
   
   print(testJacobian(net, A))
   print(testJacobianParameters(net, A))
end

function test.StereoJoin()
   A = torch.CudaTensor(6, 8, 4, 12):normal()
   n = jzt.StereoJoin(3)
   print(testJacobian(n, A))
end

function test.LinearRelu()
   A = torch.CudaTensor(5, 3):normal()

   net = nn.Sequential()
   net:add(jzt.Linear(3, 4))
   net:add(jzt.Relu())
   net = net:cuda()

   print(testJacobian(net, A))
   print(testJacobianParameters(net, A))
end

test = {}
function test.SpatialBias()
   A = torch.CudaTensor(5, 3):normal()

   net = jzt.SpatialBias()
   print(testJacobian(net, A))
   print(testJacobianParameters(net, A))
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
