%% multivariate stochastic volatility
%% with reduces sources of error

function [h0, Ah, gammah, S] = SVRSOE(erry, h0, Ah, gammah, h00, ah0, iVh00, iVah0, sharedvol )

Ystar = log( ( erry ) .^ 2 + 0.0001 );

[n, rh] = size( Ah );
Tn = length( erry ); T = Tn / n;
h = vec( repmat( h0, 1, T ) + Ah * gammah );

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

% sample h0, Ah
dconst = mi( S )';
invOmega = spdiags( 1 ./ sigi( S )', 0, Tn, Tn );

b0 = [h00; ah0( 1:n * rh )];
iVb0 = sparse( 1:n * ( rh + 1 ), 1:n * ( rh + 1 ), [iVh00; iVah0( 1:n * rh )] );

W = kron( [ones( T, 1 ) gammah'], speye( n ) );
WinvOmega = W' * invOmega;
iVb = iVb0 + WinvOmega * W;
b_hat = iVb \ ( iVb0 * b0 + WinvOmega * ( Ystar - dconst ) );
b = b_hat + chol( iVb ) \ randn( n * ( rh + 1 ), 1 );
        
h0 = b( 1:n );
Ah = reshape( b( n + 1:end ), n, rh );

% generate a draw of gammah conditional on gammax
if sharedvol < 2 && rh > 0
    bigAh = kron( speye( T ), Ah );
    bigAhinvOmega = bigAh' * invOmega;

    H = spdiags( -ones( T * rh, 1 ), -rh, speye( T * rh ) );
    iVgam = H' * H + bigAhinvOmega * bigAh;
    gam_hat = iVgam \ ( bigAhinvOmega * ( Ystar - dconst - repmat( h0, T, 1 ) ) );
    gammah = reshape( gam_hat + chol( iVgam ) \ randn( rh * T, 1 ), rh, T );
end