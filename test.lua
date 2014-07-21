require 'Test'
require 'jzt'
require 'nn'
require 'image'
require 'cunn'
require 'cutorch'

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

function test.Sqrt()
   cutorch.manualSeed(42)
   A = torch.CudaTensor(5):uniform()
   n = jzt.Sqrt():cuda()
   print(testJacobian(n, A))
end

function test.SpatialClassNLLCriterion()
   cutorch.manualSeed(1)
   A = torch.CudaTensor(1, 2, 3, 3):normal()
   t = torch.CudaTensor(1, 1, 3, 3):uniform():mul(3):floor()

   m = jzt.SpatialClassNLLCriterion()
   print(testCriterion(m, A, t))
end

function test.mask()
   A = torch.CudaTensor(1, 1, 4, 4):normal()
   T = torch.CudaTensor(1, 1, 4, 4):uniform():mul(3):floor()
   print(A)
   print(T)
   jzt.mask(A, T, A)
   print(A)
end

function test.spatial_argmin()
   A = torch.CudaTensor(1, 3, 4, 4):normal()
   O = torch.CudaTensor(1, 1, 4, 4):zero()
   jzt.spatial_argmin(A, O)
   print(A)
   print(O)
end

function test.Margin1Loss()
   A = torch.CudaTensor(1, 3, 4, 4):normal()
   T = torch.CudaTensor(1, 1, 4, 4):uniform():mul(4):floor()
   n = jzt.Margin1Loss(0.5)
   
   print(testCriterion(n, A, T))
end

function test.SpatialRandnPadding()
   A = torch.CudaTensor(2, 1, 3, 3):normal()
   n = jzt.SpatialRandnPadding(2, 2, 2, 2):cuda()
   print(testJacobian(n, A))
end

function test.Margin2Loss()
   A = torch.CudaTensor(1, 3, 4, 4):normal()
   T = torch.CudaTensor(1, 1, 4, 4):uniform():mul(4):floor()
   n = jzt.Margin2Loss(1)
   
   print(testCriterion(n, A, T))
end

function test.cbca()
   img = torch.CudaTensor(1, 1, 5, 6):normal()
   disp = torch.CudaTensor(1, 7, 5, 6):normal()
   n = jzt.CBCA(img, 0.1, 2)
   print(testJacobian(n, disp))

--   height = 250
--   width = 1242
--   n_tr = 194
--   n_te = 195
--   xl0 = torch.FloatTensor(torch.FloatStorage('/home/jure/devel/kitti/data/xl0.bin')):reshape(n_tr + n_te, 1, height, width)
--   pred = torch.FloatTensor(torch.FloatStorage('/home/jure/devel/kitti/tmp/pred_juretov_1_00001')):reshape(1, 228, height, width)
--
--   img = xl0[1]:reshape(1, 1, height, width):cuda()
--   disp = pred:cuda()
--
--   _, i = disp:float():max(2)
--   image.savePNG('foo0.png', i[{1,1}]:double():div(228))
--
--   n = jzt.CBCA(img, 0.2, 20)
--   n:forward(disp)
--
--   _, i = n.output:float():max(2)
--   image.savePNG('foo1.png', i[{1,1}]:double():div(228))
end

function test.Mul()
   A = torch.CudaTensor(6, 8, 4, 12):normal()
   n = jzt.Mul(-1):cuda()
   print(testJacobian(n, A))
end

function test.SpatialMaxout()
   A = torch.CudaTensor(8, 8, 4, 5):normal()
   n = jzt.SpatialMaxout(2)

   print(testJacobian(n, A))
end

function test.Linear1()
   A = torch.CudaTensor(3, 2):normal()
   n = jzt.Linear1(2)

   print(testJacobian(n, A))
   print(testJacobianParameters(n, A))
end

function test.SpatialConvolution1()
   A = torch.CudaTensor(1, 2, 3, 4)
   n = jzt.SpatialConvolution1(2, 5)

   print(testJacobian(n, A))
   print(testJacobianParameters(n, A))
end

function test.SpatialKernelNLLCriterion()
   A = torch.CudaTensor(1, 5, 3, 4):normal()
   target = torch.CudaTensor(1, 1, 3, 4):uniform(0, 6):floor()
   kernel = torch.CudaTensor{1,2,1}
   n = jzt.SpatialKernelNLLCriterion(kernel):cuda()

   print(testCriterion(n, A, target))
end

function test.StereoJoin()
   A = torch.CudaTensor(6, 8, 4, 12):normal()
   n = jzt.StereoJoin(3, 'L1')
   print(testJacobian(n, A))
end

test = {}
function test.StereoJoin2()
   disp_max = 228 
   A = torch.CudaTensor(12, 32, 32, 228 + 32):normal()
   B = A:double()
   n = jzt.StereoJoin2(disp_max):cuda()

   n:forward(A)
   print(n.output:size())

   left  = B:index(1, torch.range(1,A:size(1),2):long())
   right = B:index(1, torch.range(2,A:size(1),2):long())

   C = torch.Tensor(A:size(1)/2, disp_max, A:size(3), A:size(4) - disp_max):zero()
   for i = 1,C:size(2) do
      ll = left:narrow(4,disp_max+1,C:size(4))
      rr = right:narrow(4,disp_max+1-i+1,C:size(4))
      C[{{},i,{},{}}]:copy(torch.add(ll, -1, rr):abs():sum(2))
   end
   print(C:add(-1, n.output:double()):abs():max())

end


for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
