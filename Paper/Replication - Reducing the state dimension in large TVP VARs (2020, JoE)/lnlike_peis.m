function lnlike = lnlike_peis( diffy, XA, Ah, h0, H, gam_hat, iVgam, nparts )
% Particle Efficient Importance Sampling procedure based on Algorithm 2 of
% Scharth and Kohn (2013)

% setup constants
[n, r] = size( Ah );
T = length( diffy ) / n;

% assume we already have an "efficient importance density"
% gamma ~ N( gam_hat, iVgam^{-1} ), where gam_hat maximizes
% p( gamma ) * p( y | gamma ) and iVgam is the negative Hessian evaluated
% at the moment, i.e. iVgam = H' * H + blkdiag( D );
% beware: the below procedure will not work unless all above is satisfied

% first, we need to construct the sequential densities
% q( gamma_1 | y ); q( gamma_t | gamma_{t-1}, y ) from
% q( gamma | y ) by a backwards recursion
C = cell( T, 1 ); err = cell( T, 1 );
Vt = iVgam( end - r + 1:end, end - r + 1:end );
C{T} = Vt - speye( r );
err{T} = chol( C{T} + speye( r ) ) \ randn( r, nparts / 2 );
err{T} = [err{T} -err{T}]; % augment with antithetic draws
for t = 1:T - 1
	Vt = iVgam( r * ( T - t - 1 ) + 1:r * ( T - t ),  ...
                    r * ( T - t - 1 ) + 1:r * ( T - t ) ) - Vt \ speye( r );
	C{T - t} = Vt - speye( r );
    err{T - t} = chol( Vt ) \ randn( r, nparts / 2 );
    err{T - t} = [err{T - t} -err{T - t}]; % augment with antithetic draws
end
b = ( H + blkdiag( C{:} ) ) * gam_hat;

% Note: b, C parameterize the sequential densities as:
% gamma_1 ~ N( (C_1 + I)^{-1}*b_1, (C_1 + I)^{-1} );
% gamma_t | gamma_{t-1} ~ N( (C_t + I)^{-1}*(b_t + gamma_{t-1}, (C_t + I)^{-1} )

% now sample / compute likelihood contribution with forward recursion
lnlike = 0;
gamma0 = zeros( r, nparts );
lnWt = repmat( -log( nparts ), 1, nparts );
resamp = false;
for t = 1:T
    % sample gamma_1 | y_1 and compute the weights
    ridxt = r * ( t - 1 ) + 1:r * t;
    nidxt = n * ( t - 1 ) + 1:n * t;
	gamma1 = ( C{t} + speye( r ) ) \ bsxfun( @plus, b( ridxt ), gamma0 ) + err{t};

    lnwt = lnWt + ...
           + lnft( diffy( nidxt ), XA( nidxt, ridxt ), Ah, h0, gamma1, gamma0 ) ...
           - lnkappa( b( ridxt ), C{t}, gamma1, gamma0 );
    
    if resamp
        mlnwt = mean( lnwt );
        lnlike = lnlike + log( sum( exp( lnwt - mlnwt ) ) ) + mlnwt ...
                        + log( sum( exp( lnfwt - mlnfwt ) ) ) + mlnfwt;
    else
        lnwt = lnwt + lnchi( b( ridxt ), C{t} + speye( r ), gamma0 );
        mlnwt = mean( lnwt );
        lnlike = lnlike + log( sum( exp( lnwt - mlnwt ) ) ) + mlnwt;
    end
    lnWt = lnwt - log( sum( exp( lnwt - mlnwt ) ) ) - mlnwt;
    if t < T
        lnfwt = lnWt + lnchi( b( ridxt + r ), C{t + 1} + speye( r ) , gamma1 );
        mlnfwt = mean( lnfwt );
        lnfWt = lnfwt - log( sum( exp( lnfwt - mlnfwt ) ) ) - mlnfwt;
        lnESS = -log( sum( exp( 2 * lnfWt ) ) );
        
        if lnESS - log( nparts ) < log( 0.5 )
            % resample the draws with duplication
            gamma0( :, 1:nparts / 2 ) = gamma1( :, resampleSystematic( exp( lnfWt ), nparts / 2 ) );
            gamma0( :, nparts / 2 + 1:end ) = gamma0( :, 1:nparts / 2 );
            lnWt = repmat( -log( nparts ), 1, nparts );
            resamp = true;
        else
            gamma0 = gamma1;
            resamp = false;
        end 
    end
end
end

function out = lnft( yt, XtA, Ah, h0, x1, x0 )
    [n, r] = size( Ah );
    ht = bsxfun( @plus, h0, Ah * x1 );
    lnprior = -0.5 * ( r * log( 2 * pi ) + sum( ( x1 - x0 ) .^ 2, 1 ) );
    lnlike = -0.5 * ( n * log( 2 * pi ) + sum( ht + exp( -ht ) .* bsxfun( @minus, yt, XtA * x1 ) .^ 2, 1 ) );
    out = lnprior + lnlike;
end
    
function out = lnkappa( bt, Ct, x1, x0 )
    r = size( x1, 1 );
    out = -0.5 * ( r * log( 2 * pi ) + sum( ( x1 - x0 ) .^ 2, 1 ) ) ...
          + bt' * x1 - 0.5 * sum( x1 .* ( Ct * x1 ), 1 );
end

function out = lnchi( bt, iVt, x0 )
    choliVt = chol( iVt, 'lower' );
    out = -sum( log( diag( choliVt ) ) ) ...
          -0.5 *  sum( x0 .^ 2 - ( choliVt \ bsxfun( @plus, bt, x0 ) ) .^ 2, 1 );
end