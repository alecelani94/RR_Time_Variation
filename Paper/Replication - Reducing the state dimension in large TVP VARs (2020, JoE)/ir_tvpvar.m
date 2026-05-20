function IR = ir_tvpvar( s_B, s_A, hrz, tper,  resp, shocks, disprog )
% Compute impulse responses for the TVP-VAR model:
%
% y_t = B_{1,t} y_{t-1} + ... + B_{p,t} y_{t-p} + A_t e_t,  e_t ~ N(0,I_n)
%
% for a horizon of 'hrz' periods.

n = size( s_B, 1 );
p = size( s_B, 2 ) / n;
nsims = size( s_B, 4 );

if nargin < 5
    % by default, compute all responses to all shocks and show progress
    resp = 1:n;
    shocks = 1:n;
    disprog = true;
elseif nargin < 6
    shocks = 1:n;
    disprog = true;
elseif nargin < 7
    disprog = true;
end

% allocate space
IR = zeros( length( resp ), length( shocks ), 1 + hrz, length( tper ), nsims );

if disprog
    disp( 'Simulating impulse responses...' );
    txtlen = 0; tic;
end
for isim = 1:nsims
    for t = tper
        A_t = s_A( :, :, t, isim );
        B = s_B( :, :, :, isim );
        Cs = eye( n * p );
        IR( :, :, 1, tper == t, isim ) = A_t( resp, shocks );
        for s = 1:hrz
            Cs = [B( :, :, t + s ); [eye( n * ( p - 1 ) ) zeros( n * ( p - 1 ), n )]] * Cs;
            IR( :, :, 1 + s, tper == t, isim ) = Cs( resp, 1:n ) * A_t( :, shocks );
        end
    end
    
    if disprog
        % show progress
        txtlen = showprog( isim, nsims, 100, txtlen );
    end
end

if disprog
    fprintf( '\nIR simulation completed after %5.3f minutes.\n', toc / 60 );
end