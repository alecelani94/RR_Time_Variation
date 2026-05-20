function IR = irf( hrz, tper, resp, shocks, s_theta0, s_Ax, s_gammax, s_h0, s_Ah, s_gammah, T, n, k, p )

svsims = size( s_theta0, 2 );
IR = zeros( length( resp ), length( shocks ), 1 + hrz, length( tper ), svsims );

disp( 'Simulating impulse responses...' );
txtlen = 0; tic;
for isim = 1:svsims
    % recover B_{0t}...B_{pt}, h
    theta = bsxfun( @plus, s_theta0( :, isim ), s_Ax( :, :, isim ) * s_gammax( :, :, isim ) );
    h = bsxfun( @plus, s_h0( :, isim ), s_Ah( :, :, isim ) * s_gammah( :, :, isim ) );
    
    B0 = triu( ones( n, n ) );
    B0idx = [B0 - B0' - eye( n ); zeros( k, n )]; B0idx = B0idx( B0idx > -1 );
    
    B0 = repmat( B0, [1 1 T] );
    B0( B0 - repmat( eye( n ), [1 1 T] ) > 0 ) = theta( B0idx == 1, : );
    
    B0 = permute( B0, [2 1 3] );
    B1 = permute( reshape( theta( B0idx == 0, : ), k, n, T ), [2 1 3] );
    
    for t = tper
        Gt = B0( :, :, t ) \ eye( n ); %diag( exp( h( :, t ) / 2 ) );
        Cs = eye( n * p );
        IR( :, :, 1, tper == t, isim ) = Gt( resp, shocks );
        for s = 1:hrz
            Cs = [B0( :, :, t + s ) \ B1( :, k - n * p + 1:end, t + s ); ...
                 [eye( n * ( p - 1 ) ) zeros( n * ( p - 1 ), n )]] * Cs;
            IR( :, :, 1 + s, tper == t, isim ) = Cs( resp, shocks ) * Gt( resp, shocks );
        end
    end
    
    % show progress
    txtlen = showprog( isim, svsims, 100, txtlen );
end
fprintf( '\nIR simulation completed after %5.3f minutes.\n', toc / 60 );