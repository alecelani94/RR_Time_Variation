function scales = AR_scales(Data,P)

[Y,X] = VAR_objects(Data,P);

[T,N] = size(Y);

scales = zeros(N,1);
for i = 1:N

    y = Y(:,i);

    x = X(:,[1,1+(i:N:P*N)]);

    b = (x'*x)\(x'*y);

    err = y-x*b;

    scales(i) = (err'*err) / T;
end