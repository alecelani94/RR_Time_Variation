function [scales, Sigma_AR] = AR_scales(Data, P)
% AR_SCALES  AR(P) residual variances (and full residual covariance) per
% equation, using intercept + own lags only.
%
%   scales   : N x 1 -- per-equation residual variance
%   Sigma_AR : N x N -- full cross-equation residual covariance, useful as
%              an IW center for Sigma (E[Sigma] = Sigma_AR when scale =
%              (nu - N - 1)*Sigma_AR).

[Y, X] = VAR_objects(Data, P);
[T, N] = size(Y);

scales = zeros(N, 1);
E      = zeros(T, N);
for i = 1:N
    y = Y(:, i);
    x = X(:, [1, 1+(i:N:P*N)]);
    b = (x'*x) \ (x'*y);
    err = y - x*b;
    scales(i) = (err'*err) / T;
    E(:, i)   = err;
end

if nargout > 1
    Sigma_AR = (E' * E) / T;
end