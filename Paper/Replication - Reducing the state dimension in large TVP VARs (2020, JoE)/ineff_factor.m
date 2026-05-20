function [ef, L] = ineff_factor( X, L )

n = size( X, 1 );
ef = zeros( n, 1 );

if nargin < 2
    L = zeros( n, 1 );
end

for i = 1:n
    [ef( i ), L( i )] = ineff_factor1( X( i, : ), L( i ) );
end

end

function [ef1, L] = ineff_factor1( x, L )

if L == 0
    L = max( get_L( x ), 20 );
end

%%% debug %%%
% [r, l, b] = autocorr( x, length( x ) - 1, L );
% [L b( 1 ) r( L ) r( L + 1 )], autocorr( x, L + 10, L );
%%%%%%%%%%%%%

r = autocorr( x, L );
ef1 = 1 + 2 * sum( r( 2:end ) );

% r = acorr_simple( x, L );
% w = 1 - ( 1:L ) / ( L + 1 );
% ef1 = 1 + 2 * sum( w' .* r( 2:end ) );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = acorr_simple( x, L )

r = zeros( L, 1 );

for i = 1:L
	y = x( 1:( end - i ) );
	z = x( ( 1 + i ):end );
	r( i ) = mean( ( y - mean( x ) ) .* ( z - mean( x ) ) );
end

r = [1; r / var( x, 1 )];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function L = get_L( x )

L = 0;
n = length( x );

[r, l, b] = autocorr( x, n - 1, L );
L_new = min( find( abs( r ) < b( 1 ) ) ) - 1;

i = 1;
while i <= 100 && L ~= L_new
	L = L_new;
	[r, l, b] = autocorr( x, L );
	L_new = min( find( abs( r ) < b( 1 ) ) ) - 1;
    i = i + 1;
end

if L ~= L_new
	warning( 'Could not find appropriate truncation lag.  Setting to 0.' );
	L = 0;
end

end