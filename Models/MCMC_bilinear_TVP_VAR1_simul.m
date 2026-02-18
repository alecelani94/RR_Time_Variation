function [Phi_bar_draws, B_draws, a_draws, c_draws, sigma2_draws] = MCMC_bilinear_TVP_VAR1_simul(data, lambda, gamma, par_aux, init, burn, mcmc, thin, print)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Gibbs sampler for Bilinear TVP-VAR(1) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% - Model:
%   y(t) = Phi(t) * y_(t-1) + e(t), e(i,t) ~ N(0, sigma2(i))
% 
% - Parameter decomposition:
%   Phi(t) = Phi_bar + A(t) * B * C(t)
%   A(t)   = diag(a(t)) 
%   C(t)   = diag(c(t))
% 
% - Dimensions:
%   Phi_bar, B : [N x N]
%   a(t), c(t) : [N x 1]
%
% - Inputs:
%   Y      : [T x N] data matrix
%   lambda : [2 x 1] vector of prior variances for Phi_bar and B
%   gamma  : [2 x 1] vector of IG parameters for sigma2

%% dimensions %%

[T, N] = size(data);

Teff = T - 1;

%% prior section %%

%%% mean, static coefficients %%%

% bar_phi ~ N(0, lambda(1)*I)
iV_phi_diag = (1/lambda(1)) * eye(N);

% vec(B') ~ N(0, lambda(2)*I)
iV_B_diag = (1/lambda(2)) * eye(N);

%%% mean, dynamic coefficients %%%

ar1_aux    = par_aux(:,1);
sigma2_aux = par_aux(:,2);

% lag matrix: L * x stacks x_{t-1} with x_1 = 0
L = [sparse(N, N*Teff); speye(N*(Teff-1)) sparse(N*(Teff-1), N)];

% a(t): (I - rho_a * L) a = eps_a,  eps_a ~ N(0, sigma2_a * I)
%   rho_a = 1 => RW,  rho_a < 1 => AR(1)
H_a  = speye(N*Teff) - ar1_aux(1) * L;

% c(t): (I - rho_c * L) c = eps_c,  eps_c ~ N(0, sigma2_c * I)
H_c  = speye(N*Teff) - ar1_aux(2) * L;

%%% variances %%%

% gamma = [shape_e, mean_e]
%   sigma2(i) ~ IG(a_e, b_e),  mean = b_e/(a_e-1) = gamma(2)

a_e = gamma(1);
b_e = (a_e - 1) * gamma(2);

%% define objects %%

Y = data(2:end,:);
X = data(1:end-1,:);

%% initialization %%

% initialize from true paramaters for simplicity

Phi_bar = init.Phi_bar; % N x N
B       = init.B;       % N x N
A       = init.a;       % T x N
C       = init.c;       % T x N
sigma2  = init.sigma2;  % N x 1

%% storing %%

Phi_bar_draws = zeros(mcmc, N, N);
B_draws       = zeros(mcmc, N, N);
a_draws       = zeros(mcmc, T, N);
c_draws       = zeros(mcmc, T, N);
sigma2_draws  = zeros(mcmc, N);

%% precompute %%

XX      = X'*X;
HH_a    = (1/sigma2_aux(1)) * (H_a' * H_a);
HH_c    = (1/sigma2_aux(2)) * (H_c' * H_c);
D_x     = spdiags(vec(X'), 0, N*Teff, N*Teff);   % Block 4: X is fixed
I_Teff  = speye(Teff);                           % Block 4: kron base
a_post  = a_e + Teff / 2;                        % Block 5: IG shape
isigma2 = 1 ./ sigma2;

Y_tilde = zeros(Teff, N);
E       = zeros(Teff, N);

%% ---- Gibbs sampler --------------------------------------------------------

niter = burn + mcmc * thin;

s = 1;
tic;
for iter = 1:niter


    %% block 1: Phi_bar %%

    for t = 1:Teff
        Y_tilde(t,:) = Y(t,:) - X(t,:) * (B .* (A(t+1,:)'*C(t+1,:)))';
    end

    for i = 1:N
        post_prec = iV_phi_diag + XX / sigma2(i);
        post_rhs  = X' * Y_tilde(:,i) / sigma2(i);
        Phi_bar(i,:) = draw_precision(post_rhs, post_prec)';
    end

    Y_bar      = Y - X * Phi_bar';
    vec_Y_bar  = vec(Y_bar');

    %% block 2: B %%

    for i = 1:N
        Xtilde_i  = A(2:T,i) .* (C(2:T,:) .* X);
        post_prec = iV_B_diag + (Xtilde_i' * Xtilde_i) / sigma2(i);
        post_rhs  = Xtilde_i' * Y_bar(:,i) / sigma2(i);
        B(i,:) = draw_precision(post_rhs, post_prec)';
    end

    %% block 3: a(t) %%

    Xtilde        = (C(2:T,:) .* X) * B';
    sigma_inv_mat = repmat(isigma2, Teff, 1);
    vec_Xtilde    = vec(Xtilde');

    post_prec = HH_a + spdiags(vec_Xtilde.^2 .* sigma_inv_mat, 0, N*Teff, N*Teff);
    post_rhs  = vec_Xtilde .* vec_Y_bar .* sigma_inv_mat;

    A(2:T,:) = reshape(draw_precision(post_rhs, post_prec), N, Teff)';

    %% block 4: c(t) %%

    Xtilde = spdiags(vec(A(2:T,:)'), 0, N*Teff, N*Teff) ...
           * kron(I_Teff, B) * D_x;

    S_inv     = spdiags(sigma_inv_mat, 0, N*Teff, N*Teff);
    post_prec = HH_c + Xtilde' * S_inv * Xtilde;
    post_rhs  = Xtilde' * (sigma_inv_mat .* vec_Y_bar);

    C(2:T,:) = reshape(draw_precision(post_rhs, post_prec), N, Teff)';

    %% block 5: sigma2 %%

    for t = 1:Teff
        E(t,:) = Y_bar(t,:) - X(t,:) * (B .* (A(t+1,:)' * C(t+1,:)))';
    end

    for i = 1:N
        sigma2(i) = igamrnd(a_post, b_e + sum(E(:,i).^2) / 2);
    end

    isigma2 = 1 ./ sigma2;

    %% store %%
    if iter > burn && mod(iter - burn, thin) == 0

        Phi_bar_draws(s,:,:) = Phi_bar;
        B_draws(s,:,:)       = B;
        a_draws(s,:,:)       = A;
        c_draws(s,:,:)       = C;
        sigma2_draws(s,:)    = sigma2';

        s = s+1;
    end

    if mod(iter, print) == 0
        fprintf('Iteration %d / %d\n', iter, niter);
    end
end
fprintf('Elapsed time: %.2f secs\n', toc);


