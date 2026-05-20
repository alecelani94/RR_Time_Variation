function [gstar, iVgam, conv, grad] = gam_mode( diffy, XA, Ah, h0, H, g1 )
warning off MATLAB:singularMatrix;
warning off MATLAB:nearlySingularMatrix;

linesearch = false;
maxLStries = 1; % setting 1 means no line search
maxtries = 1000; % this for the main Newton loop
[n, r] = size( Ah );
T = length( diffy ) / n;

if nargin < 6 || isempty( g1 )
    g1 = zeros( r * T, 1 );
end
conv = true;

HH = H' * H;
h = repmat( h0, 1, T ) + Ah * reshape( g1, r, T );
ibigSig = spdiags( exp( -h( : ) ), 0, n * T, n * T );
erry = diffy - XA * g1;
zeta = spdiags( erry, 0, n * T, n * T ) * kron( speye( T ), Ah ) + XA;
D1 = -HH * g1 - 0.5 * repmat( sum( Ah, 1 )', T, 1 ) + 0.5 * ( zeta + XA )' * ibigSig * erry;
F = -HH - 0.5 * kron( speye( T ), Ah' * Ah ) - XA' * ibigSig * XA;
d = -F \ D1;

tries = 0;
while ~( norm( d ) < 1e-3 ) && tries <= maxtries - 1
    g0 = g1;
    if tries > 0
        % save last known values in case we run into problem at this iter
        D10 = D1;
        D20 = D2_0;
    end
    
    lamL = 0; lamU = 2; h_lamC = Inf; LStries = 0;
    while abs( lamU - lamL ) > 1e-12 && abs( h_lamC ) > 1e-2 && LStries <= maxLStries - 1
        lamC = ( lamL + lamU ) / 2;
        g1 = g0 + lamC * d;
        
        h = repmat( h0, 1, T ) + Ah * reshape( g1, r, T );
        ibigSig = spdiags( exp( -h( : ) ), 0, n * T, n * T );
        sqrtibigSig = spdiags( exp( -h( : ) / 2 ), 0, n * T, n * T );
        erry = diffy - XA * g1;
        zeta = spdiags( erry, 0, n * T, n * T ) * kron( speye( T ), Ah ) + XA;
        D1 = -HH * g1 - 0.5 * repmat( sum( Ah, 1 )', T, 1 ) + 0.5 * ( zeta + XA )' * ibigSig * erry;
        h_lamC = D1' * d;
        if h_lamC < 0
            lamU = lamC;
        else
            lamL = lamC;
        end
        LStries = LStries + 1;
    end

    H_sqrtibigSig_zeta_XA = [H; sqrtibigSig * zeta / sqrt( 2 ); sqrtibigSig * XA / sqrt( 2 )];
    zXAiSigXA = ( zeta - XA )' * ibigSig * XA;
    D2_0 = H_sqrtibigSig_zeta_XA' * H_sqrtibigSig_zeta_XA;
    D2_1 = ( zXAiSigXA + zXAiSigXA' ) / 2;
    iVgam = D2_0 + D2_1;
    
    if linesearch
        d = -D1;
    else
        d = D2_0 \ D1;
    end

    if any( ~isfinite( d ) )
        tries = maxtries;
        g1 = g0;
        D1 = D10;
        iVgam = D20;
        break;
    end
    
    tries = tries + 1;
end

% finalize
gstar = g1;
grad = D1;

if tries == maxtries
    % didn't converge, so return the best value found
    fprintf( '\nFull Newton-Raphson failed to converge.\n\n' );
    conv = false;
end

warning on MATLAB:singularMatrix;
warning on MATLAB:nearlySingularMatrix;
end

function out = f( diffy, XA, Ah, h0, x  )

[n, r] = size( Ah );
T = length( diffy ) / n;
H = spdiags( -ones( T * r, 1 ), -r, speye( T * r ) );

out = -0.5 * ( r * T * log( 2 * pi ) + sum( ( H * x ) .^ 2 ) ) ...
                   - 0.5 * ( n * T * log( 2 * pi ) + sum( vec( repmat( h0, 1, T ) ...
                   + Ah * reshape( x, r, T ) ) ...
                   + exp( -vec( repmat( h0, 1, T ) + Ah * reshape( x, r, T ) ) ) ...
                   .* ( diffy - XA * x ) .^ 2 ) );
end