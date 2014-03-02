-- B = softmax(A), x is temporary storage
function jzt.softmax(A, B, x)
   jzt.max(A, x, 1)
   jzt.sub_mat_vect(A, x, B, 1)
   jzt.exp(B, B)
   jzt.sum(B, x, 1)
   jzt.div_mat_vect(B, x, B, 1)
end

function jzt.sub(x, y, z)
   jzt.add(x, y, z, -1)
end
