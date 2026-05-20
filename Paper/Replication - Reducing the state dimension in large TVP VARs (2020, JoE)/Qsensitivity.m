rxx = 2:6;
run = 55;
load 'results_n15_rx2_rh3_share1_run55.mat';
[nk, ~, svsims] = size( s_Ax );

s_Q = repmat( {zeros( nk, nk, svsims )}, length( rxx ), 1 );
mQ = repmat( {zeros( nk, nk )}, length( rxx ), 1 );

for ii = 1:length( rxx )
    if ii > 1
        load( ['results_n' num2str( n ) '_rx' num2str( rxx( ii ) ) ...
            '_rh' num2str( rh ) '_share' num2str( sharedvol ) '_run' ...
            num2str( run ) '.mat'] );
    end
    
    for isim = 1:svsims
        s_Q{ii}( :, :, isim ) = s_Ax( :, :, isim ) * s_Ax( :, :, isim )';
    end
    mQ{ii} = median( s_Q{ii}, 3 );
end

delta = repmat( {nan( length( rxx ), length( rxx ) )}, 5, 1 );

for ii = 1:length( rxx )
    for jj = 1:length( rxx )
        delta{1}( ii, jj ) = sum( diag( ...
            ( eye( nk ) + mQ{ii} ) / ( eye( nk ) + mQ{jj} ) ) ) / nk;
        delta{2}( ii, jj ) = norm( mQ{ii} - mQ{jj} );
        delta{3}( ii, jj ) = sum( abs( diag( mQ{ii} - mQ{jj} ) ) );
        delta{4}( ii, jj ) = norm( mQ{ii} - mQ{jj}, 'fro' );
        delta{5}( ii, jj ) = 1 - sum( diag( mQ{ii} * mQ{jj} ) ) ...
            / norm( mQ{ii}, 'fro' ) / norm( mQ{jj}, 'fro' );
    end
end