function [G, C] = blkdrep( F, n, T )

jmax = floor( log( T ) / log( 2 ) );
rest = T - 2 ^ jmax;

[d1, m] = size( F );

if n > d1
    error( 'Stagger step exceeds block size.' );
end

% 1st block
% if isempty( F )
%     Graw = A';
% else
%     Graw = [A'; cell2mat( mat2cell( F', n, repmat( n, p, 1 ) )' )];
% end

% middle blocks
Graw = F;

for j = 1:jmax
    w1 = n * 2 ^ ( j - 1 );
    w2 = m * 2 ^ ( j - 1 );
    sp = sparse( w1, w2 );
    Graw = [[Graw; sp] [sp; Graw]];
end

% rest of blocks
if rest > 0
    w1 = n * 2 ^ jmax;
    w2 = m * 2 ^ jmax;
    z1 = n * rest;
    z2 = m * rest;
    G2 = Graw( 1:( z1 + d1 - n ), 1:z2 );
    sp1 = sparse( z1, w2 );
    sp2 = sparse( w1, z2 );
    Graw = [[Graw; sp1] [sp2; G2]];
end

G = Graw( 1:( end - d1 + n ), : );

% x = size( Graw, 1 ) - size( G, 1 );
C = Graw( ( end - ( d1 - n ) + 1 ):end, ( end - ( d1 - n ) / n * m + 1 ):end );