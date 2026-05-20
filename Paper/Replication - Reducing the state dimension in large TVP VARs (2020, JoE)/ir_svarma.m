function IR = ir_svarma( s_B, s_A, s, diff2var, disprog )

if nargin < 5
    % by default, show progress
    disprog = true;
end

% Compute impulse responses for the VARMAX model:
%
% y_t = B_1 y_{t-1} + ... + B_p y_{t-p} + A_0 e_t,  e_t ~ N(0,I_n)
%
% for a horizon of 's' periods.

n = size( s_B, 1 );
p = size( s_B, 2 ) / n;
q = size( s_A, 2 ) / n - 1;
nsims = size( s_B, 3 );

% allocate space
IR = zeros( n, s + 1, n, nsims );

if disprog
    disp( 'Simulating impulse responses...' );
    txtlen = 0; tic;
end
for i = 1:nsims
    if nargin < 4
        Bblk = cell2mat( mat2cell( s_B( :, :, i ), n, repmat( n, 1, p ) )' );
    else
        Bblk = cell2mat( mat2cell( [-eye( n ) s_B( :, :, i )] * diff2var, n, repmat( n, 1, p + 1 ) )' );
    end
    Ablk = cell2mat( mat2cell( s_A( :, :, i ), n, repmat( n, 1, q + 1 ) )' );
    K = blkdrep( [speye( n ); -Bblk], n, s + 1 ) \ [Ablk; zeros( ( s - q ) * n, n )];
    IR( :, :, :, i ) = reshape( K, [n s + 1 n] );
    
    if disprog
        % show progress
        txtlen = showprog( i, nsims, 100, txtlen );
    end
end

if disprog
    fprintf( '\nIR simulation completed after %5.3f minutes.\n', toc / 60 );
end