function [B0, VB0] = minnprior( Y, p )
% setup Minnesota prior for
%    y(t) = c + B1 * y(t-1) + ... + Bp * y(t-p) + e(t),
% where the in puts are:
%    Y: (n x T) matrix [y(1)...y(T)],
%    B: (n x n + n * p) matrix [c,B1,...,Bp]
%    p: number of lags

% extract dimensions
[n, T] = size( Y );

Y0 = Y( :, 1:p );  % store the first p obs as init cond
shortY = Y( :, ( p + 1 ):end, : );
T = T - p;

X0 = zeros( n * p, T );
for j = 1:p
    X0( ( j - 1 ) * n + 1:j * n, : ) = [Y0( :, p - j + 1:end ) shortY( :, 1:T - j )];
end
X = [ones( 1, T ); X0];
k = size( X, 1 );

% tunning constants
% pi1 = 1; pi2 = 0.3 / pi1 / n; pi3 = 2;
pi1 = 1 / ( 2 * n ); pi2 = 0.3; pi3 = 2;

% compute scales from residuals of univariate regressions
s2 = diag( shortY * ( eye( T ) - X' / ( X * X' ) * X ) * shortY' / ( T - k ) );
B0 = zeros( n, n, p ); VB0 = zeros( n, n, p );
for i = 1:n
    for j = 1:n
        for l = 1:p
            if i == j
                VB0( i, j, l ) = pi1 / l ^ pi3;
                if l == 1
                    B0( i, j, l ) = 1;
                end
            else
                VB0( i, j, l ) = pi1 * pi2 * s2( i ) / s2( j ) / l ^ pi3;
            end
        end
    end
end
