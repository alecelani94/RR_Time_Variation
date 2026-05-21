function IFs = compute_IFs(draws)
% COMPUTE_IFS  Inefficiency factors for every field of an MCMC draws struct.
%
%   draws.<field> is expected to have its first dimension equal to the
%   number of stored draws Mc (chain length). For each field, the IF is
%   computed independently for every cell of the remaining dimensions:
%
%     draws.Psi_bar  [Mc x N x K]        -> IFs.Psi_bar  [N x K]
%     draws.theta    [Mc x M x T]        -> IFs.theta    [M x T]
%     draws.A        [Mc x T x N x R]    -> IFs.A        [T x N x R]
%     draws.lambda_* [Mc x 1]            -> IFs.lambda_* scalar
%
%   So for TV parameters the IF is reported per (parameter, t).

IFs = struct();
fn  = fieldnames(draws);
for f = 1:numel(fn)
    A   = draws.(fn{f});
    sz  = size(A);
    Mc  = sz(1);
    if numel(sz) == 2 && sz(2) == 1
        IFs.(fn{f}) = ineff_factor(A);
    else
        rest        = sz(2:end);
        Aflat       = reshape(A, Mc, []);
        IF_flat     = ineff_factor(Aflat);
        IFs.(fn{f}) = reshape(IF_flat, [rest 1]);
    end
end
end
