function IF = ineff_factor(X)
% INEFF_FACTOR  Inefficiency factor IF = 1/ESS_frac for each column of X.
%
%   IF = ineff_factor(X) where X is Mc x d, returns a d x 1 vector of
%   inefficiency factors. IF = integrated autocorrelation time, i.e.
%   IF = 1 + 2 * sum_k rho_k with Geyer's initial-positive-sequence
%   truncation (see ess.m / autocorr_fft.m).
%
%   For a scalar chain pass a column vector.

d  = size(X, 2);
IF = zeros(d, 1);
for j = 1:d
    e = ess(X(:, j));
    if e <= 0
        IF(j) = NaN;        % no positive-sequence truncation found
    else
        IF(j) = 1 / e;
    end
end
end
