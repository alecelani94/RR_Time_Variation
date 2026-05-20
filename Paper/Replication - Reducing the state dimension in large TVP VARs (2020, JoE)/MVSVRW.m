%% multivariate stochastic volatility
%% with random walk transition eq

function [h S] = MVSVRW(Ystar,h,iSig,iVh)

n = size(iSig,1);
Tnh = size(h,1);
Tn = size(Ystar,1);

if Tnh > Tn
    % sampling h0, h
    h0 = h( 1:n );
    h = h( n + 1:end );
end

%% normal mixture
pi = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mi = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigi = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigi = sqrt(sigi);

%% sample S from a 7-point distrete distribution
temprand = rand(Tn,1);
q = repmat(pi,Tn,1).*normpdf(repmat(Ystar,1,7),repmat(h,1,7)+repmat(mi,Tn,1), repmat(sqrtsigi,Tn,1));
q = q./repmat(sum(q,2),1,7);
S = 7 - sum(repmat(temprand,1,7)<cumsum(q,2),2)+1;
    
%% sample h
% y^* = h + d + \epison, \epison \sim N(0,\Omega),
% Hh = \alpha + \nu, \nu \ sim N(0,S),
% where d_t = Ez_t, \Omega = diag(\omega_1,\ldots,\omega_n), 
% \omega_t = var z_t, S = diag(Dh/(1-\phi^2), sig, \ldots, sig)
if Tnh > Tn
    Hh =  spdiags( -ones( Tnh, 1 ), -n, speye( Tnh ) ); %%%speye(Tn) - spdiags(ones(Tn,1),-n,Tn,Tn);
    invSh = blkdiag( iVh, kron( speye( Tn / n ), iSig ) ); %%%spdiags([iVh; 1/sig*ones(Tn-1,1)],0,Tn,Tn);
    dconst = mi(S)'; invOmega = spdiags(1./sigi(S)',0,Tn,Tn);
    Kh = Hh'*invSh*Hh;
    Xh = sparse( 1:Tn, n + 1:Tnh, ones( Tn, 1 ), Tn, Tnh );
    Ph = Kh + Xh' * invOmega * Xh;
    Ch = chol(Ph);
    hhat = Ph\(Xh' * invOmega*(Ystar-dconst));
    h = hhat + Ch\randn(Tnh,1);
else
    Hh =  spdiags( -ones( Tn, 1 ), -n, speye( Tn ) ); %%%speye(Tn) - spdiags(ones(Tn,1),-n,Tn,Tn);
    invSh = blkdiag( iVh, kron( speye( Tn / n - 1 ), iSig ) ); %%%spdiags([iVh; 1/sig*ones(Tn-1,1)],0,Tn,Tn);
    dconst = mi(S)'; invOmega = spdiags(1./sigi(S)',0,Tn,Tn);
    Kh = Hh'*invSh*Hh;
    Ph = Kh + invOmega;
    Ch = chol(Ph);
    hhat = Ph\(invOmega*(Ystar-dconst));
    h = hhat + Ch\randn(Tn,1);
end
