function jzt.savePNG(fname, tensor, jet)
   local n_images, height, width

   if tensor:dim() == 2 then
      n_images = 1
      height = tensor:size(1)
      width = tensor:size(2)
   elseif tensor:dim() == 3 then
      n_images = tensor:size(1)
      height = tensor:size(2)
      width = tensor:size(3)
   elseif tensor:dim() == 4 then
      n_images = tensor:size(1)
      height = tensor:size(3)
      width = tensor:size(4)
   end

   local x = torch.Tensor(n_images, height, width):copy(tensor)
   for i = 1,n_images do
      x[i]:add(-x[i]:min()):div(x[i]:max() - x[i]:min())
   end

   x:resize(n_images * height, width)

   if jet then
      col = torch.Tensor(3, n_images * height, width)
      jzt.grey2jet(x, col)
      x = col
   end

   image.savePNG(fname, x)
end

-- B = softmax(A), x is temporary storage
function jzt.softmax(A, B, x)
   jzt.max(A, x, 2)
   jzt.sub_mat_vect(A, x, B, 2)
   jzt.exp(B, B)
   jzt.sum(B, x, 2)
   jzt.div_mat_vect(B, x, B, 2)
end

function jzt.sub(x, y, z)
   jzt.add(x, y, z, -1)
end
