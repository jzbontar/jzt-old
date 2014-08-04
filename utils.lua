
------------------------------

function tdims(t)
   local dims = ''
   if t == nil then
      dims = '(no tensor)'
   else
      if #t:size() == 1 then
	 dims = string.format('%d', t:size(1))
      elseif #t:size() == 2 then
	 dims = string.format('%dx%d', t:size(1), t:size(2))
      elseif #t:size() == 3 then
	 dims = string.format('%dx%dx%d', t:size(1), t:size(2), t:size(3))
      elseif #t:size() == 4 then
	 dims = string.format('%dx%dx%dx%d', t:size(1), t:size(2), t:size(3), t:size(4))
      end
   end
   return dims
end

-- returns a string with available memory for current cuda device
function cuda_mem_str(msg)
   if not cutorch then return '' end
   if not msg then msg = ''
   else msg = msg .. ' - ' end
   local s = msg
   local pipe = io.popen('nvidia-smi -a | grep \"Name\\|Used\\|Free\\|Gpu\" | grep \"[^A]$\" | grep \" *\" | grep Free')
   if pipe then
      local lines = pipe:lines()
      if lines then
	 local j = 1
	 for i in lines do
	    i = string.split(i, ':')
	    if j == cutorch.getDevice() then
	       s = s .. ' device ' .. j .. ':' .. i[2]
	    end
	    j = j + 1
	 end
      end
   end
   return s
end

-- returns a string with information about a tensor t
function tinfo_cuda(t)
   if t == nil then return "(no tensor)" end
   if type(t) == 'table' then
      print(t)
      return ''
   end
   local mean
   local type = 'no_type'
   if t.type then type = t:type()
   else print(t) end
   if t:nDimension() == 0 then return "(empty tensor)"
   elseif not t.mean then mean = t:float():mean()
   else  mean = t:mean() end
   return string.format('(%s min: %f mean: %f max: %f type: %s cuda: %s)',
			tdims(t), t:min(), mean, t:max(), type, cuda_mem_str())
end

function module_name(m, i)
   return string.format('%02d_%s', i, torch.typename(m))
end

-- reshape tensor so that trailing dimensions of 1 disappear
-- e.g. 128x100x1x1 -> 128x100
function remove_trailing_dims(t)
   local n = t:dim()
   local i = t:dim()
   while i > 1 do
      if t:size(i) == 1 then n = n - 1
      else i = 0 end
      i = i - 1
   end
   if n > 0 then
      s = torch.LongStorage(n)
      for i=1,n do s[i] = t:size(i) end
      t = t:resize(s)
   end
   return t
end

-- insert n dimensions of size 1 at offset o
function size_insert_dims(t, n, o)
   local dims
   if torch.typename(t) == 'torch.LongStorage' then dims = t
   else dims = t:size() end
   o = o or 1
   -- insert new dimensions
   local s = torch.LongStorage(dims:size() + n)
   local off = 1
   for i = 1,dims:size()+n do
      if i >= o and i < o + n then s[i] = 1
      else
	 s[i] = t[off]
	 off = off + 1
      end
   end
   return s
end

-- remove n dims starting at offset o provided that theses dims are
-- of size 1, and return the new size (but not reshaping tensor)
function size_remove_dims(t, n, o)
   o = o or 1
   -- insert new dimensions
   local s = torch.LongStorage(t:dim() - n)
   local off = 1
   for i = 1,t:dim() do
      if not(i >= o and i < o + n) then
	 s[off] = t:size(i)
	 off = off + 1
      end
   end
   return s
end

-- insert n dimensions of size 1 at offset o
function insert_dims(t, n, o)
   -- reshape: use resize instead because reshape not available in cuda
   -- resize is ok since we're not changing the number of elements
   return t:resize(size_insert_dims(t, n, o))
end

-- remove n dims starting at offset o provided that theses dims are
-- of size 1, return the reshaped tensor
function remove_dims(t, n, o)
   -- reshape: use resize instead because reshape not available in cuda
   -- resize is ok since we're not changing the number of elements
   return t:resize(size_remove_dims(t, n, o))
end

-- resize all tensor's dimensions to 1
function resize1(t)
   if t then
      if type(t) == 'table' then
	 for i=1,#t do t[i] = resize1(t[i]) end
      else
	 local sz = t:size()
	 for i = 1,sz:size() do sz[i] = 1 end
	 t = t:resize(sz)
      end
   end
   return t
end

-- replace tensor with tensor with same order but dimensions of 1
function replace1(t, replace_tables)
   if t then
      if type(t) == 'table' then
	 if replace_tables then
	    for i=1,#t do t[i] = replace1(t[i]) end
	 end
      else
	 local sz = t:size()
	 for i = 1,sz:size() do sz[i] = 1 end
	 t = torch.Tensor(sz)
      end
   end
   return t
end

