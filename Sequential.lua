-- dependencies
local here = paths.dirname(paths.thisfile())
dofile(here .. '/utils.lua')

local Sequential, parent = torch.class('jzt.Sequential', 'nn.Module')

function Sequential:__init(arg)
   self.debug = 0
   self.timing = 0
   self.bprop_min = 0
   if arg then
      self.debug = arg.debug or 0
      self.timing = arg.timing or 0
      -- minimum layer id to bprop to, 0: bprop to l1 inputs only
      -- 1: bprop l2 inputs and l1 weights
      self.bprop_min = arg.bprop_min or 0
      -- only fprop up until this id (included)
      self.fprop_max = arg.fprop_max
      -- if true, do not keep outputs of modules
      self.fprop_only = arg.fprop_only
      if arg.name then self.name = arg.name .. ' '
      else self.name = '' end
   end
   self.modules = {}
end

function Sequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function Sequential:size()
   return #self.modules
end

function Sequential:get(index)
   return self.modules[index]
end

function Sequential:updateOutput(input)
   if self.debug == 1 then print(self.name..'-> input: ' .. tinfo_cuda(input)) end
   local time
   if self.timing == 1 then time = sys.clock() end
   local currentOutput = input
   for i=1,#self.modules do
      if not self.fprop_max or i <= self.fprop_max then
	 local name = module_name(self.modules[i], i)
	 if self.debug == 1 then
	    print(self.name..'-> input:' .. tinfo_cuda(currentOutput)..'\t'..name)
	 end
	 if self.timing == 1 then time = sys.clock() end

	 currentOutput = self.modules[i]:updateOutput(currentOutput)
	 -- clean up module output if only fproping
	 if self.fprop_only then
	    input = nil
	    self.modules[i].output = replace1(self.modules[i].output)
	    collectgarbage()
	 end

	 if self.timing == 1 then
	    cutorch.synchronize() -- forces to wait
	    if not self.ftime then self:reset_timing() end
	    self.ftime[name] = self.ftime[name]+sys.clock()-time
	 end
      end
   end

   self.output = currentOutput

   if self.prop_only then
      self.output = replace1(self.output)
      collectgarbage()
   end

   if self.debug == 1 then print(self.name..'-> output: ' .. tinfo_cuda(self.output)) end
   return currentOutput
end

-- function Sequential:updateOutput(input)
--    local currentOutput = input
--    for i=1,#self.modules do 
--       currentOutput = self.modules[i]:updateOutput(currentOutput)
--    end 
--    self.output = currentOutput
--    return currentOutput
-- end

function Sequential:updateGradInput(input, gradOutput)
   local time
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      if i >= self.bprop_min then
	 local name = module_name(currentModule, i+1)
	 if self.debug == 1 then
	    print(self.name..'<- gradOutput: ' .. tinfo_cuda(currentGradOutput)..'\t'..name)
	 end

	 local previousModule = self.modules[i]
	 if self.timing == 1 then time = sys.clock() end
	 currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
	 if self.timing == 1 then
	    cutorch.synchronize() -- forces to wait
	    self.btime[name] = self.btime[name]+sys.clock()-time
	 end
	 currentModule = previousModule
      end
   end
   if self.bprop_min == 0 then
      if self.timing == 1 then time = sys.clock() end
      currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
      if self.timing == 1 or self.debug then
	 local name = module_name(currentModule, 1)
	 if self.timing == 1 then
	    cutorch.synchronize() -- forces to wait
	    self.btime[name] = self.btime[name]+sys.clock()-time
	 end
	 if self.debug == 1 then
	    local name = module_name(currentModule, 1)
	    print(self.name..'<- gradOutput: ' .. tinfo_cuda(currentGradOutput)..'\t'..name)
	 end
      end
   end
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local time
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      if i+1 >= self.bprop_min then
	 local name = module_name(currentModule, i+1)
	 if self.debug == 1 then
	    print(self.name..'<a gradOutput: ' .. tinfo_cuda(currentGradOutput)..'\t'..name)
	 end

	 local previousModule = self.modules[i]
	 if self.timing == 1 then time = sys.clock() end
	 currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
	 currentGradOutput = currentModule.gradInput
	 if self.timing == 1 then
	    cutorch.synchronize() -- forces to wait
	    self.atime[name] = self.atime[name]+sys.clock()-time
	 end
	 currentModule = previousModule
      end
   end
   if self.bprop_min <= 1 then
      if self.timing == 1 then time = sys.clock() end
      currentModule:accGradParameters(input, currentGradOutput, scale)
      if self.debug == 1 then
	 local name = module_name(currentModule, 1)
	 print(self.name..'<a gradOutput: ' ..tinfo_cuda(currentGradOutput)..'\t'..name)
      end
      if self.timing == 1 then
	 cutorch.synchronize() -- forces to wait
	 local name = module_name(currentModule, 1)
	 self.atime[name] = self.atime[name]+sys.clock()-time
      end
   end
end

function Sequential:accUpdateGradParameters(input, gradOutput, lr)
   local time
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if self.timing == 1 then time = sys.clock() end
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      if self.timing == 1 then
	 cutorch.synchronize() -- forces to wait
	 local name = module_name(currentModule, i+1)
	 self.utime[name] = self.utime[name]+sys.clock()-time
      end
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   if self.timing == 1 then time = sys.clock() end
   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
   if self.timing == 1 then
      cutorch.synchronize() -- forces to wait
      local name = module_name(currentModule, 1)
      self.utime[name] = self.utime[name]+sys.clock()-time
   end
end

function Sequential:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function Sequential:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function Sequential:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Sequential:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

local function tinsert(to, from)
   if type(from) == 'table' then
      for i=1,#from do
	 tinsert(to,from[i])
      end
   else
      table.insert(to,from)
   end
end

-- add parameters to the parameters to be returned by
-- parameters()
function Sequential:add_parameters(w, gw)
   print('adding '..#w..' extra parameters to sequential.')
   if not self.extra_parameters then self.extra_parameters = {} end
   if not self.extra_parameters.w then self.extra_parameters.w = {} end
   if not self.extra_parameters.gw then self.extra_parameters.gw = {} end
   tinsert(self.extra_parameters.w, w)
   tinsert(self.extra_parameters.gw, gw)
end

function Sequential:parameters()
   if self.shared then return end
   local w = {}
   local gw = {}
   if self.extra_parameters then
      if self.extra_parameters.w then tinsert(w, self.extra_parameters.w) end
      if self.extra_parameters.gw then tinsert(gw, self.extra_parameters.gw) end
   end
   for i=1,#self.modules do
      if not self.modules[i].shared then
	 local mw,mgw = self.modules[i]:parameters()
	 if mw then
	    tinsert(w,mw)
	    tinsert(gw,mgw)
	 end
      end
   end
   return w,gw
end

function Sequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'jzt.Sequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      local m = self.modules[i]
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(m):gsub(line, line .. tab)
      if m.weight then str = str..line..tab..tab..tab..'weights: '..tinfo_cuda(m.weight) end
      if m.bias then str = str..line..tab..tab..tab..'biases: '..tinfo_cuda(m.bias) end
   end
   str = str .. line .. '}'
   return str
end


function Sequential:reset_timing()
   self.ftime = {}
   self.btime = {}
   self.atime = {}
   self.utime = {}
   self.ttime = {}
   for i=1,#self.modules do
      local m = self.modules[i]
      local name = module_name(m, i)
      self.ftime[name] = 0
      self.btime[name] = 0
      self.atime[name] = 0
      self.utime[name] = 0
      self.ttime[name] = 0
      if torch.typename(m) == 'jzt.Sequential' then m:reset_timing() end
   end
end

function Sequential:print_timing(iter)
   iter = iter or 1
   if self.timing == 1 then
      print('WARNING: cuda synchronization between modules is on')
      local total = 0
      local total_forward = 0
      local total_backward = 0
      local total_acc = 0
      for i=1,#self.modules do
	 local name = module_name(self.modules[i], i)
	 local c = 1000 / iter
	 local f = self.ftime[name] * c
	 local b = self.btime[name] * c
	 local a = self.atime[name] * c
	 local u = self.utime[name] * c
	 local t = f + b + a + u
	 total = total + t
	 total_forward = total_forward + f
	 total_backward = total_backward + b
	 total_acc = total_acc + a
      end
      if iter > 1 then print('(averaged over '..iter..' iterations)') end
      print('total (ms) ' .. total..' forward '..total_forward
	 ..' backward '..total_backward..' acc '..total_acc)
      print('total%\ttotal\tfprop\tbprop\tacc\tupdate')
      for i=1,#self.modules do
	 local name = module_name(self.modules[i], i)
	 local c = 1000 / iter
	 local f = self.ftime[name] * c
	 local b = self.btime[name] * c
	 local a = self.atime[name] * c
	 local u = self.utime[name] * c
	 local t = f + b + a + u
	 local p = t*100/total
	 print(string.format('%.1f%%\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%s', p, t, f, b, a, u, name))
      end
   end
end

-- returns a list of sums of gradients
-- if nsamples is defined, normalize by number of samples
function Sequential:sumGradients(nsamples, verbose)
   if verbose then print('L1 of gradients per layer:') end
   local nsamples = nsamples or 1
   local l = {}
   for i = 1,#self.modules do
      local m = self.modules[i]
      local w = m.gradWeight
      local b = m.gradBias
      if w or b then
	 l[i] = { name = string.format('%02d_%s', i, torch.typename(m)) }
	 if w then l[i].weight = w:float():abs():sum() / nsamples end
	 if b then l[i].bias = b:float():abs():sum() / nsamples end
	 if verbose then
	    print(string.format('weights %f\tbiases %f\t%s',
				l[i].weight, l[i].bias, l[i].name))
	 end
      end
   end
   return l
end

-- returns a list of the sum of differences of all the weights of each layer
-- between now and the last call to this function.
function Sequential:weights_differences(verbose)
   if verbose then print('Differences per layer:') end
   local l = {}
   if not self.last_weights then self.last_weights = {} end
   if not self.last_biases then self.last_biases = {} end
   for i = 1,#self.modules do
      local m = self.modules[i]
      local w = m.weight
      local b = m.bias
      if w or b then
	 local name = module_name(m, i)
	 l[i] = { name = name }
	 if w then
	    if not self.last_weights[i] then self.last_weights[i] = w:clone():double() end
	    -- mean of absolute differences per weight
	    --print('weights: ' .. tinfo(w) .. ' ' .. name)
	    local wd = w:clone():double()
	    --print(tinfo(wd))
	    --print(tinfo(self.last_weights[i]))
	    l[i].weight = self.last_weights[i]:add(-1, wd):abs():sum() / w:nElement()
	    -- update last weights
	    self.last_weights[i] = wd
	 end
	 if b then
	    if not self.last_biases[i] then self.last_biases[i] = b:clone():double() end
	    -- mean of absolute differences per bias
	    local bd = b:clone():double()
	    l[i].bias = self.last_biases[i]:add(-1, bd):abs():sum() / b:nElement()
	    -- update last biases
	    self.last_biases[i] = bd
	 end
	 if verbose then
	    print(string.format('weights %.10f\tbiases %.10f\t%s',
				l[i].weight, l[i].bias, l[i].name))
	 end
      end
   end
   return l
end

-- multiplies gradients of each layer by a factor
function Sequential:scaleGradients(factors, verbose)
   if verbose then print('gradient factors per layer:') end
   for i = 1,#self.modules do
      local f = factors[i]
      if f then
	 local m = self.modules[i]
	 if m.gradWeight then m.gradWeight:mul(f) end
	 if m.gradBias then m.gradBias:mul(f) end
	 if verbose then
	    print('multiplying gradients of layer ' .. i .. ' by ' .. f .. ' ' .. torch.typename(m))
	 end
      end
   end
end

function Sequential:set_bprop_min(min)
   self.bprop_min = min
   print('+++ changing bprop min to ' .. min)
end

-- cleans up memory
function Sequential:clean(clean_tables)
   for i=1,#self.modules do
      self.modules[i].output = replace1(self.modules[i].output, clean_tables)
      if self.modules[i].clean then
	 self.modules[i]:clean(clean_tables)
      end
   end
   self.output = replace1(self.output, clean_tables)
   collectgarbage()
end
