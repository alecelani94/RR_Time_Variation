function [ghstar, iVgamh] = gamh_mode( diffy, XA, Ah, h0, Hh, Hx, sharedvol, gamh )

usegpu = false; % we can do some large array computation on the GPU, if availabile

[n, rh] = size( Ah );
T = length( diffy ) / n;
HHx = Hx' * Hx;
bigAh = kron( speye( T ), Ah );

if nargin < 8
    gamh = zeros( rh * T, 1 );
end

if sharedvol
    HHh = Hh' * Hh;
    [gamh, Dh1] = em_gamh( diffy, XA, Ah, h0, HHh, HHx, rh, n, T, gamh( : ) );
    h = vec( repmat( h0, 1, T ) + Ah * reshape( gamh, rh, T ) );
else
    HhAh = Hh * spdiags( repmat( 1 ./ diag( Ah ), T, 1 ), 0, rh * T, rh * T );
    HHh = HhAh' * HhAh;
    [h, Dh1] = em_h( diffy, XA, h0, HHh, HHx, n, T, repmat( h0, T, 1 ) + gamh( : ) );
    gamh = ( reshape( h, rh, T ) - repmat( h0, 1, T ) ) ./ repmat( diag( Ah ), 1, T );
    Dh1 = bigAh' * Dh1 * bigAh; % need to adjust b/c em_h works directly with 'h', hot 'gamh'
end

ghstar = vec( gamh );

% compute the remainder of the Hessian
ibigSig = spdiags( exp( -h( : ) ), 0, n * T, n * T );
XAibigSig = XA' * ibigSig;
if usegpu
    C = gpuArray( full( XAibigSig' / ( HHx + XAibigSig * XA ) * XA' ) );
    Dh2 = bigAh' * gather( C' .* ( eye( n * T ) - C ) ) * bigAh;
else
    C = XAibigSig' / ( HHx + XAibigSig * XA ) * XA';
    Dh2 = bigAh' * ( C' .* ( eye( n * T ) - C ) ) * bigAh;
end
Dh2( Dh2 < 1e-9 ) = 0;
iVgamh = -( Dh1 - sparse( Dh2 ) / 2 );
end

function [gh, D2] = em_gamh( diffy, XA, Ah, h0, HHh, HHx, rh, n, T, gh )

if nargin < 10
    gh = zeros( rh * T, 1 );
end

bigAh = kron( speye( T ), Ah );
lastg = Inf; x1 = zeros( rh * T, 1 );
ibigSig = spdiags( exp( -( repmat( h0, T, 1 ) + bigAh * gh ) ), 0, n * T, n * T );

while ~( norm( gh - lastg ) < 1e-3 )
	lastg = gh;
	XAibigSig = XA' * ibigSig;
	iVgam = HHx + XAibigSig * XA;
	gam_hat = iVgam \ ( XAibigSig * diffy );

	ciVgam = chol( iVgam );
	c = sum( ( XA / ciVgam ) .^ 2, 2 ) + ( diffy - XA * gam_hat ) .^ 2;
	x0 = Inf;
	while ~( norm( x1 - x0 ) < 1e-3 )
		x0 = x1;
		z = c .* exp( -( repmat( h0, T, 1 ) + bigAh * x0 ) );
		D1 = -HHh * x0 - 0.5 * bigAh' * ( ones( n * T, 1 ) - z );
		D2 = -HHh - 0.5 * bigAh' * spdiags( z, 0, n * T, n * T ) * bigAh;
		x1 = x0 - D2 \ D1;
	end
	gh = x1;
	ibigSig = spdiags( exp( -( repmat( h0, T, 1 ) + bigAh * gh ) ), 0, n * T, n * T );

end
end

function [h, D2] = em_h( diffy, XA, h0, HHh, HHx, n, T, h )

if nargin < 8
    h = zeros( n * T, 1 );
end

lasth = Inf; x1 = zeros( n * T, 1 );
ibigSig = spdiags( exp( -h ), 0, n * T, n * T );

while ~( norm( h - lasth ) < 1e-3 )
	lasth = h;
	XAibigSig = XA' * ibigSig;
	iVgam = HHx + XAibigSig * XA;
	gam_hat = iVgam \ ( XAibigSig * diffy );

	ciVgam = chol( iVgam );
	c = sum( ( XA / ciVgam ) .^ 2, 2 ) + ( diffy - XA * gam_hat ) .^ 2;
	x0 = Inf;
	while ~( norm( x1 - x0 ) < 1e-3 )
		x0 = x1;
		z = c .* exp( -( repmat( h0, T, 1 ) + x0 ) );
		D1 = -HHh * x0 - 0.5 * ones( n * T, 1 ) + 0.5 * z;
		D2 = -HHh - 0.5 * spdiags( z, 0, n * T, n * T );
		x1 = x0 - D2 \ D1;
	end
	h = repmat( h0, T, 1 ) + x1;
	ibigSig = spdiags( exp( -h ), 0, n * T, n * T );
end
end