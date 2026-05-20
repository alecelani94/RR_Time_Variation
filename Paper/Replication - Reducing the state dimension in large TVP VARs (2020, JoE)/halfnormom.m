function [Exm, I] = halfnormom( mu, sig2, m )

h = -mu / sqrt( sig2 );

I = [1 normpdf( h ) / ( 1 - normcdf( h ) ) zeros( 1, m - 1 )];
Exm = mu ^ m + m * mu ^ ( m - 1 ) * sqrt( sig2 ) * I( 2 );
for r = 2:m
    I( r + 1 ) = h ^ ( r - 1 ) * I( 2 ) + ( r - 1 ) * I( r - 1 );
    Exm = nchoosek( m, r ) * mu ^ ( m - r ) * sig2 ^ ( r / 2 ) * I( r + 1 );
end
    