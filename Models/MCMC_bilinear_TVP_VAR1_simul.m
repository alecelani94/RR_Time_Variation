function results = MCMC_bilinear_TVP_VAR1_simul(Data, lambda, gamma, init, Burn, MCMC, Thin)

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

[T, N] = size(Data);

Teff = T - 1;
K    = N^2;

%% prior section %%

%%% mean, static coefficients %%%

% bar_phi ~ N(0,V_phi), bar_Phi = vec(bar_Phi')
V_phi  = lambda(1) * ones(N,N);
iV_phi = 1 ./ V_phi;

% vec(B') ~ N(0, lambda(2)*I_K)

V_B  = lambda(2) * ones(N,N);
iV_B = 1 ./ V_B;

%%% mean, dynamic coefficients %%%

% a(t) - a(t-1) ~ N(0, I_N), a(1) = 0 fixed
% H_a * a(2:T) ~ N(0, I_{N(T-1)})

H_a  = speye(N*Teff) - [[sparse(N,N*(Teff-1)); speye(N*(Teff-1))] sparse(N*Teff,N)];

% c(t) - c(t-1) ~ N(0, I_N), c(1) = 0 fixed
% H_c * c(2:T) ~ N(0, I_{N(T-1)})

H_c  = speye(N*Teff) - [[sparse(N,N*(Teff-1)); speye(N*(Teff-1))] sparse(N*Teff,N)];

%%% variances %%%

% gamma = [shape_e, mean_e]
%   sigma2(i) ~ IG(a_e, b_e),  mean = b_e/(a_e-1) = gamma(2)

a_e = gamma(1);
b_e = (a_e - 1) * gamma(2);

%% define objects %%

Y = Data(2:end,:);
X = Data(1:end-1,:);

%% initialization %%

% initialize from true paramaters for simplicity


Phi_bar = init.Phi_bar;              % N x N
B       = init.B;                    % N x N
A       = init.a;                    % T x N
C       = init.c;                    % T x N
sigma2  = init.sigma2;

%% storing %%

Phi_bar_draws = zeros(MCMC, N, N);
B_draws       = zeros(MCMC, N, N);
A_draws       = zeros(MCMC, T, N);
C_draws       = zeros(MCMC, T, N);
sigma2_draws  = zeros(MCMC, N);

%% useful %%

XX = X'*X;

HH_a = H_a' * H_a;

Phi_tilde = zeros(T, N, N);
Y_tilde   = zeros(T-1, N);

%% Gibbs sampler %%

Niter = Burn + MCMC * Thin;
for iter = 1:Niter


    %% block 1: Phi_bar %%

    for t = 1:T-1

        Y_tilde(t,:) = Y(t,:) - X(t,:) * (B .* (A(t+1,:)'*C(t+1,:)) )';
    end
    % vectorized alternative (faster for large N):
    % CX = C(2:T,:) .* X;
    % Y_tilde = Y - A(2:T,:) .* (CX * B');

    for i = 1:N

        post_prec = diag(iV_phi(i,:)) + XX / sigma2(i);
        post_rhs  = X' * Y_tilde(:,i) / sigma2(i);

        Phi_bar(i,:) = draw_precision(post_rhs, post_prec)';
    end

    %% block 2: B %%

    Y_bar = Y - X * Phi_bar';

    for i = 1:N

        Xtilde_i = A(2:T,i) .* (C(2:T,:) .* X);

        post_prec = diag(iV_B(i,:)) + (Xtilde_i' * Xtilde_i) / sigma2(i);
        post_rhs  = Xtilde_i' * Y_bar(:,i) / sigma2(i);

        B(i,:) = draw_precision(post_rhs, post_prec)';
    end

    %% block 3: a(t) %%

    Xtilde = (C(2:T,:) .* X) * B';       

    sigma_inv_mat = repmat(1./sigma2, T-1, 1);

    temp = vec(Xtilde').^2 .* sigma_inv_mat;

    post_prec = HH_a + spdiags(temp, 0, N*Teff, N*Teff);
    post_rhs  = vec(Xtilde') .* vec(Y_bar') .* sigma_inv_mat;

    vec_A   = draw_precision(post_rhs, post_prec);
    A(2:T,:) = reshape(vec_A, N, T-1)';

    %% block 4: c(t) %%

    %% block 5: sigma2(i) %%

    

    %% store %%
    if iter > nburn
        s = iter - nburn;
        Phi_bar_draws(s,:) = vec(Phi_bar)';
        b_draws(s)         = b;
        Sigma_draws(s,:,:) = Sigma;
        a_post_mean        = a_post_mean + a / nsave;
        c_post_mean        = c_post_mean + c / nsave;
    end

    if mod(iter, 1000) == 0
        fprintf('MCMC iteration %d / %d\n', iter, ntot);
    end
end

%% ---- Results --------------------------------------------------------------
results.Phi_bar = Phi_bar_draws;
results.b       = b_draws;
results.Sigma   = Sigma_draws;
results.a_mean  = a_post_mean;
results.c_mean  = c_post_mean;

end

%% ---- Local helper ----------------------------------------------------------
function val = get_opt(s, fld, default)
    if isfield(s, fld)
        val = s.(fld);
    else
        val = default;
    end
end
