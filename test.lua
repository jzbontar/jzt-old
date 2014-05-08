require 'Test'
require 'jzt'
require 'nn'
require 'image'
require 'cunn'
require 'cutorch'
require 'prof-torch'

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

function test.LinearRelu()
   A = torch.CudaTensor(5, 3):normal()

   net = nn.Sequential()
   net:add(jzt.Linear(3, 4))
   net:add(jzt.Relu())
   net = net:cuda()

   print(testJacobian(net, A))
   print(testJacobianParameters(net, A))
end

function test.SpatialBias()
   A = torch.CudaTensor(1, 5, 3):normal()

   net = jzt.SpatialBias(1, 5, 3)
   print(testJacobian(net, A))
   print(testJacobianParameters(net, A))
end

function test.L2Pooling()
   torch.manualSeed(42)
   A = torch.CudaTensor(6, 8, 12, 7):normal()
   n = jzt.L2Pooling(3, 2)

   print(testJacobian(n, A))
end

function test.L1Cost()
   A = torch.CudaTensor(5, 4, 3, 3):normal()
   n = jzt.L1Cost():cuda()

   print(testCriterion(n, A))
end

function test.ConvSplit()
   x = torch.Tensor(2, 1, 250, 1242):cuda()

   n = jzt.ConvSplit(128, 10 * 3)
   m = jzt.ConvJoin(n)
   net1 = nn.Sequential()
   net1:add(nn.SpatialConvolutionRing2(1, 32, 11, 11):cuda())
   net1:add(nn.SpatialConvolutionRing2(32, 32, 11, 11):cuda())
   net1:add(nn.SpatialConvolutionRing2(32, 32, 11, 11):cuda())
   n:forward(x)
   for i = 1,10 do
      net1:forward(n.output)
      collectgarbage()
   end
   out1 = m:forward(net1.output)

--   net2 = nn.Sequential()
--   net2:add(nn.SpatialConvolution(1, 32, 11, 11))
--   net2:add(nn.SpatialConvolution(32, 32, 11, 11))
--   net2:add(nn.SpatialConvolution(32, 32, 11, 11))
--   net2:cuda()
--   for i = 1,10 do
--      print(i)
--      net2:forward(x)
--      collectgarbage()
--   end

--   net2 = nn.Sequential()
--   net2:add(nn.SpatialConvolution(1, 32, 7, 7):cuda())
--   net2:add(nn.SpatialConvolution(32, 32, 7, 7):cuda())
--   net2:add(nn.SpatialConvolution(32, 32, 7, 7):cuda())
--   net2:get(1).weight = net1:get(1).weight
--   net2:get(1).bias = net1:get(1).bias
--   net2:get(2).weight = net1:get(2).weight
--   net2:get(2).bias = net1:get(2).bias
--   net2:get(3).weight = net1:get(3).weight
--   net2:get(3).bias = net1:get(3).bias
--   out2 = net2:forward(x)
--   cutorch.synchronize()
--   prof.tic(2)
--   net2:forward(x)
--   cutorch.synchronize()
--   prof.toc(2)
end

function test.SpatialNormalization()
   A = torch.CudaTensor(3, 8, 4, 5):normal()
   n = jzt.SpatialNormalization(8)

   print(testJacobian(n, A))
end

function test.Margin1Loss()
   cutorch.manualSeed(42)
   A = torch.CudaTensor(1, 3, 3, 3):normal()
   T = torch.CudaTensor(1, 1, 3, 3):uniform():mul(3):ceil()
   n = jzt.Margin1Loss()

   print(A)
   print(T)

   n:forward(A, T)
end

function test.StereoJoin()
   A = torch.CudaTensor(6, 8, 4, 12):normal()
   n = jzt.StereoJoin(3, 'L2_square')
   print(testJacobian(n, A))
end

function test.Mul()
   A = torch.CudaTensor(6, 8, 4, 12):normal()
   n = jzt.Mul(-1)
   print(testJacobian(n, A))
end

function test.Sqrt()
   cutorch.manualSeed(42)
   A = torch.CudaTensor(5):uniform()
   n = jzt.Sqrt():cuda()
   print(testJacobian(n, A))
end

test = {}
function test.SpatialClassNLLCriterion()
   cutorch.manualSeed(1)
   A = torch.CudaTensor(1, 2, 3, 3):normal()
   t = torch.CudaTensor(1, 1, 3, 3):uniform():mul(3):floor()

   m = jzt.SpatialClassNLLCriterion()
   print(testCriterion(m, A, t))
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
