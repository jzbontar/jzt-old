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
