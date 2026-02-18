function draw = draw_precision(r, Omega)

% Draw from N(Omega^{-1} r, Omega^{-1})
%
% Inputs:
%   r     : [K x 1] precision-weighted mean (Omega * mu)
%   Omega : [K x K] posterior precision matrix (dense or sparse)
%
% Output:
%   beta  : [K x 1] draw from the posterior

L    = chol(Omega, 'lower');
mu   = L' \ (L \ r);
draw = mu + L' \ randn(size(r));
