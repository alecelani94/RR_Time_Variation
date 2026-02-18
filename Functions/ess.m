function out = ess(x)

% ESS computes effective sample size.
%   [ESS, ESSfrac] = ess(...) computes the effective sample size
%   for an MCMC sampling chain.
%
%   The input arguments are:
%       x       - [NSAMPLES x 1] vector of posterior samples
%
%   Return values:
%       ESS     - [1 x 1] effective sample size as a fraction
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

n = length(x);
s = min(n - 1, 2000);
g = autocorr_fft(x, s);
G = g(2:s-1) + g(3:s);
ix = find(G < 0);

out = 0;
if(~isempty(ix))

    k = ix(1);
    V = g(1) + 2 * sum(g(2:k));
    ACT = V / g(1);

    out = min(n / ACT, n)/n;
end
