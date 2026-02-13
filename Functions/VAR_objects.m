function [Y,X] = VAR_objects(Data,P)

[T, N] = size(Data);

Y = Data(P+1:T, :);                 % (T-P) x N
X = zeros(T-P, N*P);                % (T-P) x (N*P)

for i = 1:P
    X(:, (i-1)*N+1 : i*N) = Data(P+1-i : T-i, :);   % lag p
end

X = [ones(T-P,1), X];