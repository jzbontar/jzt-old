require 'jzt'
require 'Test'

test = {}
function test.HuberCost()
   A = torch.rand(5, 3):cuda()
   B = torch.rand(5, 3):cuda()

   assert(testCriterion(jzt.HuberCost(1), A, B) < 1e-3)
end

function test.KLDivergence()
   A = torch.rand(5, 3):cuda():log()
   B = torch.rand(5, 3):cuda():log()

   assert(testCriterion(jzt.KLDivergence(), A, B) < 1e-3)
end

function test.ClassNLLCriterion()
   A = torch.rand(5, 3):cuda()
   B = torch.eye(3):index(1, torch.LongTensor{1,2,3,1,2}):cuda()

   assert(testCriterion(jzt.ClassNLLCriterion(), A, B) < 1e-3)
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
