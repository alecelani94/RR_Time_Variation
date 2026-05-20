%% SVRSOE_commented.m
%% Stochastic Volatility with Reduced Sources of Error
%%
%% Model (Specification 2 from Chan, Eisenstat, Strachan 2020):
%%   h_t = h0 + Ah * f_{h,t}
%%   f_{h,t} = f_{h,t-1} + z_{h,t},  z_{h,t} ~ N(0, I_{rh})
%%
%% where:
%%   h_t     = n x 1 vector of log-volatilities at time t
%%   h0      = n x 1 vector of volatility levels (to be estimated)
%%   Ah      = n x rh matrix of factor loadings (to be estimated)
%%   f_{h,t} = rh x 1 vector of latent factors (to be estimated)
%%   rh      = number of factors (rh << n for dimension reduction)
%%
%% The observation equation (after KSC transformation):
%%   y*_t = log(ε_t^2 + c) = h_t + log(χ²_1) + measurement error
%%
%% Using Kim, Shephard, Chib (1998) mixture approximation:
%%   log(χ²_1) ≈ mixture of 7 normals with known (π, m, σ²)
%%
%% This gives a linear Gaussian model conditional on mixture states S.

function [h0, Ah, gammah, S] = SVRSOE_commented(erry, h0, Ah, gammah, h00, ah0, iVh00, iVah0, sharedvol)

%% =========================================================================
%% STEP 0: SETUP AND DIMENSIONS
%% =========================================================================

% Get dimensions from the factor loadings matrix
[n, rh] = size(Ah);        % n = number of variables, rh = number of factors

% Total number of observations (vectorized)
Tn = length(erry);         % Tn = n * T
T = Tn / n;                % T = number of time periods

% Construct current log-volatilities from the factor structure
% h_t = h0 + Ah * gammah_t for each t
% In vectorized form: h = [h_1; h_2; ...; h_T] where each h_t is n x 1
h = vec(repmat(h0, 1, T) + Ah * gammah);

%% =========================================================================
%% STEP 1: TRANSFORM TO LINEAR MODEL VIA KSC APPROXIMATION
%% =========================================================================

% Transform squared residuals to approximate linear observation equation
% If ε ~ N(0, exp(h)), then log(ε²) = h + log(χ²_1)
% Add small constant (0.0001) to avoid log(0)
Ystar = log(erry.^2 + 0.0001);

%% =========================================================================
%% STEP 2: KIM-SHEPHARD-CHIB MIXTURE PARAMETERS
%% =========================================================================

% 7-component normal mixture approximation to log(χ²_1)
% These are the exact values from Kim, Shephard, Chib (1998), Table 4
% The -1.2704 adjusts for E[log(χ²_1)] = -1.2704

pi_mix = [0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575];  % mixture weights
mi = [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819] - 1.2704;  % means (adjusted)
sigi = [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261];  % variances
sqrtsigi = sqrt(sigi);  % standard deviations

%% =========================================================================
%% STEP 3: SAMPLE MIXTURE INDICATORS S (DATA AUGMENTATION)
%% =========================================================================

% For each observation, compute posterior probability of each mixture component
% p(S_i = j | y*_i, h_i) ∝ π_j * N(y*_i | h_i + m_j, σ²_j)

temprand = rand(Tn, 1);  % uniform random numbers for inverse CDF sampling

% Compute unnormalized posterior probabilities for all 7 components
% q(i,j) ∝ π_j * φ((y*_i - h_i - m_j) / σ_j)
q = repmat(pi_mix, Tn, 1) .* ...
    normpdf(repmat(Ystar, 1, 7), ...           % y*_i
            repmat(h, 1, 7) + repmat(mi, Tn, 1), ...  % h_i + m_j
            repmat(sqrtsigi, Tn, 1));          % σ_j

% Normalize to get proper probabilities (rows sum to 1)
q = q ./ repmat(sum(q, 2), 1, 7);

% Sample S via inverse CDF method
% S_i = min{j : cumsum(q_i,1:j) > u_i}
S = 7 - sum(repmat(temprand, 1, 7) < cumsum(q, 2), 2) + 1;

%% =========================================================================
%% STEP 4: SETUP FOR SAMPLING [h0; vec(Ah)]
%% =========================================================================

% Given the sampled mixture states S, the observation equation becomes:
%   y*_i - m_{S_i} = h_i + ε*_i,  where ε*_i ~ N(0, σ²_{S_i})
%
% In matrix form for the factor structure:
%   y* - d = W * b + ε*
% where:
%   d = [m_{S_1}; ...; m_{S_Tn}]  (mixture means)
%   b = [h0; vec(Ah)]             (parameters to estimate)
%   W maps b to h                 (design matrix)
%   ε* ~ N(0, Ω)                  (Ω = diag(σ²_{S_i}))

% Extract mixture means and build precision matrix for the sampled S
dconst = mi(S)';                                    % Tn x 1 vector of means
invOmega = spdiags(1 ./ sigi(S)', 0, Tn, Tn);      % Tn x Tn diagonal precision

%% =========================================================================
%% STEP 5: SAMPLE [h0; vec(Ah)] JOINTLY VIA BAYESIAN LINEAR REGRESSION
%% =========================================================================

% Stack prior parameters: b = [h0; vec(Ah)]
b0 = [h00; ah0(1:n*rh)];                            % prior mean

% Prior precision (diagonal)
iVb0 = sparse(1:n*(rh+1), 1:n*(rh+1), [iVh00; iVah0(1:n*rh)]);

% Build design matrix W
% For each t: h_t = h0 + Ah * gammah_t = [I_n, gammah_t' ⊗ I_n] * [h0; vec(Ah)]
% Stacking over t: W = kron([ones(T,1), gammah'], I_n)
W = kron([ones(T, 1), gammah'], speye(n));

% Posterior precision and mean (standard Bayesian linear regression)
% Posterior: b | y*, S ~ N(b_hat, V_b)
% V_b^{-1} = V_0^{-1} + W' Ω^{-1} W
% b_hat = V_b * (V_0^{-1} * b0 + W' Ω^{-1} * (y* - d))

WinvOmega = W' * invOmega;
iVb = iVb0 + WinvOmega * W;                         % posterior precision
b_hat = iVb \ (iVb0 * b0 + WinvOmega * (Ystar - dconst));  % posterior mean

% Sample from posterior using precision (Cholesky of precision, not covariance)
% b ~ N(b_hat, iVb^{-1})  is equivalent to  b = b_hat + chol(iVb) \ z, z ~ N(0,I)
b = b_hat + chol(iVb) \ randn(n*(rh+1), 1);

% Extract h0 and Ah from the sampled vector
h0 = b(1:n);                                        % first n elements
Ah = reshape(b(n+1:end), n, rh);                    % remaining elements reshaped

%% =========================================================================
%% STEP 6: SAMPLE gammah (FACTORS) VIA PRECISION SAMPLER
%% =========================================================================

% Only sample gammah if:
%   - sharedvol < 2 (not using shared factors with mean equation)
%   - rh > 0 (there are factors to sample)

if sharedvol < 2 && rh > 0

    % Build block-diagonal matrix for Ah across all time periods
    % bigAh = diag(Ah, Ah, ..., Ah)  (T blocks)
    bigAh = kron(speye(T), Ah);
    bigAhinvOmega = bigAh' * invOmega;

    % Build first-difference matrix H for random walk prior
    % H * gammah = [γ_1; γ_2 - γ_1; γ_3 - γ_2; ...]
    % Prior: γ_t = γ_{t-1} + z_t, z_t ~ N(0, I)
    % This implies: H * γ ~ N(0, I)  so  γ ~ N(0, (H'H)^{-1})
    H = spdiags(-ones(T*rh, 1), -rh, speye(T*rh));

    % Posterior precision for gammah
    % Combines RW prior (H'H) with likelihood (bigAh' * invOmega * bigAh)
    iVgam = H' * H + bigAhinvOmega * bigAh;

    % Posterior mean
    % Observation: y* - d - h0 = bigAh * gammah + ε*
    gam_hat = iVgam \ (bigAhinvOmega * (Ystar - dconst - repmat(h0, T, 1)));

    % Sample from posterior using precision sampler
    gammah = reshape(gam_hat + chol(iVgam) \ randn(rh*T, 1), rh, T);

end

%% =========================================================================
%% OUTPUT SUMMARY
%% =========================================================================
% Returns:
%   h0     - updated draw of volatility level (n x 1)
%   Ah     - updated draw of factor loadings (n x rh)
%   gammah - updated draw of factors (rh x T), or unchanged if sharedvol >= 2
%   S      - mixture component indicators for KSC approximation (Tn x 1)

end