function Xmap = margmode( X )

szX = size( X );
d = length( szX );

k = prod( szX( 1:d - 1 ) );
X = reshape( X, k, szX( d ) );
Xmap = zeros( k, 1 );

for i = 1:k
    [f, xi] = ksdensity( X( i, : )' );
    [~, imap] = max( f );
    Xmap( i ) = xi( imap );
end

if d > 2
    Xmap = reshape( Xmap, szX( 1:d - 1 ) );
end