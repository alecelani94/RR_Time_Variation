function [s_Dy, s_ln_prior, DIC1, p_D1, DIC2, p_D2, DIC3, p_D3, seDIC3] = rsoe_in_loop( rmaxx, rmaxh, sharedvol, doparallel )

% Program parameters
scale_data = 1;  % standardize all series to have std. dev. of 1
center_data = 1; % standardize all series to have mean 0
vecm_form = 1;   % estimate using a pseudo-VECM form (i.e. VECM with full rank Pi)
th0_prior = 0;   % type of prior for theta0:
                 % 0 - baseline normal prior (no shrinkage)
                 % 1 - lasso shrinkage prior
                 % 2 - SSVS shrinkage prior
th0_minn = 1;    % set 1 to add minnesota prior shrinkage (regardless of th0_prior setting)
armh = 1;        % use ARMH to draw gamma (only for sharedvol = 2)
mvtprop = 0;     % use a multivariate t proposal instead of normal (only for sharedvol = 2)
forceid = 0;     % transform draws to an identified parameter space in the course
                 % of the sampler; note that unidentified parameter draws
                 % can always be post-processed with the function trans2id()

% algorithm-specific controls
nsims = 20000; % desired number of MCMC simulations
burnin = .2 * nsims; % burn-in simulations to discard
progstep = 100; % frequency of progress updates
                % (i.e. every "progstep" simulations)

% set this to save every "simstep"^th draw;
% useful for running long chains on
% windows machines with limited memory
simstep = 20;
svsims = floor( nsims / simstep );

% initialize the random number generator
% note: this is the new RNG in Matlab and may not work in older versions
rng( 'shuffle' ); rngstate = rng; % a new state
% load rngstate-001; rng( rngstate ); % a previously saved state

% TVP-VAR parameters
p = 2; % number of AR lags

% MCMC tuning parameters
nu = 1e15; % tuning parameter for the multivariate t proposal
ARMHadj = 010; % tuning parameter for setting the ARMH constant;
               % large value --> longer to get an AR draw, but better
               % chances of acceptance in MH step
maxARtries = 1000; % maximum AR attempts for the ARMH algorithm

% Ydata = load( 'data_Q.csv' );
% Ydata = Ydata( :, 2:3 ); % n = 2
% Ydata = Ydata( :, 1:3 ); % n = 3
% Ydata = Ydata( :, [1 2 14 3 5 6 16] ); % n = 7
% Ydata = Ydata( :, [1 2 8 11 14 19 20 21 3 5 6 16] ); % n = 12
% Ydata = Ydata( :, [1 2 8 10 11 12 14 19 20 21 3 5 6 16 17] ); % n = 15
% I0 = true( 1, 15 ); % indicate which variables are stationary for scaling purposes
% DI0 = true( 1, 15 ); % indicate which variables are stationary for reporting purposes

Ydata = load( 'newsshocks01.csv' );
Ydata = Ydata( :, 1:15 );
I0 = false( 1, size( Ydata, 2 ) ); % indicate which variables are stationary for scaling purposes
DI0 = false( 1, size( Ydata, 2 ) ); % indicate which variables are stationary for reporting purposes

% convert to pseudo-VECM form
if vecm_form
    Ydata0 = Ydata; % save the original set (presumably in levels)
    Ylevel = Ydata( 1:end - 1, : );
    Ydata = diff( Ydata );
    I0 = true( 1, size( Ydata, 2 ) ); % indicate which variables are stationary
    p = p - 1;
else
    % do nothing
    Ylevel = [];
end
% Ydata = diff( Ydata ); I0 = true( 1, size( Ydata, 2 ) );
[T, n] = size( Ydata );

rmaxh = rmaxh + ( rmaxh == -1 ) * n; % if not set above, let it be full
if sharedvol == 0
    % for baseline SV, set rh = n
    rmaxh = n;
elseif sharedvol > 1
    % for error sharing, always force rh = rx
    rmaxx = rmaxx + rmaxh; % share all errors
    rmaxh = rmaxx;
end

% standardize the data if so desired
dscale( I0 ) = 1 + scale_data .* ( std( Ydata( :, I0 ), 1 ) - 1 );
dscale( ~I0 ) = 1 + scale_data .* ( std( diff( Ydata( :, ~I0 ) ), 1 ) - 1 );
dcenter = center_data * mean( Ydata, 1 );
Ydata = ( Ydata  - repmat( dcenter, T, 1 ) ) ./ repmat( dscale, T, 1 );

Y0 = Ydata( 1:p, : )';  % store the first p obs as init cond
shortY = Ydata( ( p + 1 ):end, : )';

T = T - p;
y = shortY( : );

X0 = zeros( n * p, T );
for j = 1:p
    X0( ( j - 1 ) * n + 1:j * n, : ) = [Y0( :, p - j + 1:end ) shortY( :, 1:T - j )];
end

if vecm_form
    Ylevel = Ylevel( p + 1:end, : );
    dscale_lev = std( Ylevel, 1 ); Ylevel = Ylevel ./ repmat( dscale_lev, T, 1 );
    dcenter_lev = mean( Ylevel, 1 ); Ylevel = Ylevel - repmat( dcenter_lev, T, 1 );
    Xdata = [ones( T, 1 ) Ylevel]'; % constant with lagged levels
else
    Xdata = ones( 1, T ); % just a constant
end
X0 = [Xdata; X0];

% define useful constants
k = n * p + size( Xdata, 1 );
nk = 0.5 * n * ( n - 1 ) + n * k; % size of parameters vector

% define useful sorting indices
freeth = [tril( true( n, n ), -1 ) true( n, k )];
mat2vec = vec( reshape( 1:n * ( n + k ), n, n + k )' );
mat2vec = mat2vec( freeth' );
[~, vec2mat] = sort( mat2vec );
[~, vec2mat2vec] = sort( vec2mat );

% construct a compact X matrix
X = nan( n - 1 + k, n * T );
Xones = cell( n, 1 );
% freeax1 = true( nk, 1 ); j = 1;
for i = 1:n
	X( n - i + 1:n - 1, i:n:end ) = -shortY( 1:i - 1, : );
	X( n:end, i:n:end ) = X0;
	Xones{i} = ones( k + i - 1, 1 );

%     if vecm_form
%         freeax1( j + i:j + i - 1 + n ) = false;
%         j = j + i + n * ( p + 1 );
%     end
end
Xones = blkdiag( Xones{:} );
% freeax2 = [true( n ^ 2, 1 ); true( n, 1 ); false( n ^ 2, 1 ); true( n ^ 2 * p, 1 )];
% norm( freeax1 - freeax2( mat2vec ) ), pause

[xrow, xcol] = find( repmat( Xones, 1, T ) );
bigX = sparse( xrow, xcol, X( ~isnan( X ) ), n * ( k + ( n - 1 ) / 2 ), n * T )';
bigXcells = mat2cell( bigX, repmat( n, T, 1 ), nk );
hugeX = blkdiag( bigXcells{:} );

% priors
ax0 = zeros( nk * rmaxx, 1 ); iVax0 = repmat( ( 2 * n ) ^ 2, nk * rmaxx, 1 );
sh0 = repmat( 16, n, 1 ); Sh0 = 0.001 * ( sh0 - 1 );
h00 = zeros( n, 1 ); iVh00 = repmat( 1, n, 1 );
ah0 = zeros( n * rmaxh, 1 ); iVah0 = repmat( ( 2 * n ) ^ 2, n * rmaxh, 1 );

if th0_minn
    % add the minnesota shrinkage on theta0 priors
    if vecm_form
        VTh00_1 = [repmat( 1, n, n + 1 ) ...                % correlations and intercept
                   repmat( ( 1 / n ) ^ 2, n, n )];          % Pi coefficients
        [~, VTh00_2] = minnprior( shortY, p );              % AR coefficients
        VTh00 = [VTh00_1 reshape( VTh00_2, n, n * p )];
        iVth00 = 1 ./ VTh00( mat2vec );
        th00 = zeros( nk, 1 );
    else
        VTh00_1 = repmat( 1, n, n + 1 );                    % correlations and intercept
        [Th00_2, VTh00_2] = minnprior( shortY, p );         % AR coefficients
        VTh00 = [VTh00_1 reshape( VTh00_2, n, n * p )];
        iVth00 = 1 ./ VTh00( mat2vec );
        Th00 = [zeros( n, k - n * ( p - 1 ) ) reshape( Th00_2, n, n * p )];
        th00 = Th00( mat2vec );
    end
else
    % just do standard normal priors
    th00 = zeros( nk, 1 ); iVth00 = repmat( 1, nk, 1 );
end
lam20( 1 ) = 1; lam20( 2 ) = 0.1; % for the lasso priors
c0 = 1e-2; q0 = 0.5; iVth00_0 = iVth00; % for the SSVS priors

% starting values
rx = rmaxx;
rh = rmaxh;
gammax = zeros( rx, T );
gammah = zeros( rh, T );

h0 = zeros( n, 1 );
Ah = zeros( n, rh );
h = zeros( n, T );
ibigSig = spdiags( exp( -h( : ) ), 0, speye( n * T ) );
Sigh = diag( Sh0 ./ ( sh0 - 1 ) );
S = zeros( n * T, 1 );

accept_gam = 0;

% allocate space for draws
% NOTE: be selective here when running on PC with low memory
s_theta0 = zeros( nk, svsims );
s_iVth00 = zeros( nk, svsims );
s_h0 = zeros( n, svsims );
s_Ax = zeros( nk, rmaxx, svsims );
s_Ah = zeros( n, rmaxh, svsims );
s_gammax = zeros( rmaxx, T, svsims );
s_gammah = zeros( rmaxh, T, svsims );
s_theta = zeros( nk, T, svsims );
s_h = zeros( n, T, svsims );
s_sigh = zeros( n, svsims );
s_S = zeros( n * T, svsims );
s_rx = zeros( 1, svsims );
s_rh = zeros( 1, svsims );
s_accept_gam = zeros( 1, svsims );

% final initializations
disp( ['Sampling ' num2str( burnin + nsims ) ' draws from the posterior...'] );

% the following is needed ONLY on Windows
% to display progress properly; on Linux
% it has no effect (but should be left defined)
txtlen = 0;

tic; % start the clock
for isim = 1:( burnin + nsims )
    % sample r
    % not yet implemented
    
    % sample Ax, Ah, gamma
    if sharedvol < 3
        % setup zero restrictions on Ax
        freeax = true( nk * rx, 1 );
        freeb = true( nk * ( rx + 1 ), 1 );
        if vecm_form && rx > 1
            % limit the time-variation in Pi to be driven by a single error
            freeAx = true( nk, rx );
%             freeAx( nk - n ^ 2 * ( p + 1 ) + 1:nk - n ^ 2 * p, 2:end ) = false;
%             freeAx( nk - n ^ 2 * ( p + 1 ) + 1:nk - n ^ 2 * p, : ) = false;
            freeax = vec( freeAx( vec2mat2vec, : ) );
            freeb = [true( nk, 1 ); freeax];
        end
%         clear freeAx;
%         freeBx = reshape( freeax, nk, rx );
%         freeAx = freeBx( vec2mat, : ); pause

        [wrow, wcol] = find( repmat( Xones, rx, T ) );
        Wsmall = kron( gammax, ones( k + n - 1, n ) ) .* repmat( X, rx, 1 );
        W = [bigX sparse( wrow, wcol, Wsmall( ~isnan( Wsmall ) ), rx * nk, n * T )'];
        W = W( :, freeb );
        
        b0 = [th00; ax0( freeax )];
        iVb0 = sparse( 1:nnz( freeb ), 1:nnz( freeb ), [iVth00; iVax0( freeax )] );
        
        WibigSig = W' * ibigSig;
        iVb = iVb0 + WibigSig * W;
        b_hat = iVb \ ( iVb0 * b0 + WibigSig * y );
        b = b_hat + chol( iVb ) \ randn( nnz( freeb ), 1 );
        
        theta0 = b( 1:nk );
        ax = zeros( nk, rx );
        ax( freeax ) = b( nk + 1:end );
        Ax = reshape( ax, nk, rx );
        
%         % alternative: sample Ax | theta0, then theta0 | Ax
%         W = sparse( wrow, wcol, Wsmall( ~isnan( Wsmall ) ), rx * nk, n * T )';
%         W = W( :, freeax );
% 
%         WibigSig = W' * ibigSig;
%         iVax = sparse( 1:nnz( freeax ), 1:nnz( freeax ), iVax0( freeax ) ) + WibigSig * W;
%         ax_hat = iVax \ ( iVax0( freeax ) .* ax0( freeax ) + WibigSig * ( y - bigX * theta0 ) );
%         shortax = ax_hat + chol( iVax ) \ randn( nnz( freeax ), 1 );
%         ax = zeros( nk, rx );
%         ax( freeax ) = shortax;
%         Ax = reshape( ax, nk, rx );
% 
%         bigXibigSig = bigX' * ibigSig;
%         iVth0 = sparse( 1:nk, 1:nk, iVth00 ) + bigXibigSig * bigX;
%         th0_hat = iVth0 \ ( iVth00 .* th00 + bigXibigSig * ( y - W * ax ) );
%         theta0 = th0_hat + chol( iVth0 ) \ randn( nk, 1 );
        
        if th0_prior == 1
            % do lasso on iVth00
            lam2_th = gamrnd( lam20( 1 ) + nk, 1 / ( lam20( 2 ) + sum( 1 ./ iVth00 ) / 2 ) );
            iVth00 = igrand( sqrt( lam2_th ) ./ abs( theta0 - th00 ), lam2_th );
        elseif th0_prior == 2
            % do SSVS on iVth00
            diff_th = theta0 - th00;
            pr = [( 1 - q0 ) ./ sqrt( c0 ./ iVth00_0 ) .* exp( -0.5 * diff_th .^ 2 ./ ( c0 ./ iVth00_0 ) ) ...
            q0 .* sqrt( 1 ./ iVth00_0 ) .* exp( -0.5 * diff_th .^ 2 .* iVth00_0 )];
            pr = pr ./ repmat( sum( pr, 2 ), 1, 2 );
                
            iVth00 = 1 ./ ( ( c0 + ( rand( nk, 1 ) < pr( :, 2 ) ) * ( 1 - c0 ) ) ./ iVth00_0 );
        end
    
        if sharedvol < 2
            % no MH step here
            accept_gam = 1;
            
            % sample gammax
            diffy = y - bigX * theta0;
            XA = hugeX * kron( speye( T ), Ax );
            XAibigSig = XA' * ibigSig;

            if rx > 0
                H = spdiags( -ones( T * rx, 1 ), -rx, speye( T * rx ) );
                iVgam = H' * H + XAibigSig * XA;
                gam_hat = iVgam \ ( XAibigSig * diffy );
            
                gammax = reshape( gam_hat + chol( iVgam ) \ randn( rx * T, 1 ), rx, T );
            end
            
            erry = diffy - XA * gammax( : );
            
            if sharedvol == 0
                % sample h0, h
                h = reshape( MVSVRW( log( erry .^ 2 + 0.0001 ), vec( [h0 h] ), inv( Sigh ), speye( n ) ), n, T + 1 );
                h0 = h( :, 1 ); h = h( :, 2:end ); % MVSVRW actually returns [h0 h]
                ibigSig = spdiags( exp( -h( : ) ), 0, speye( n * T ) );
    
                errh = h - [h0 h( :, 1:end - 1 )];
                sseh = errh * errh';
                Sigh = diag( 1 ./ gamrnd( sh0 + T / 2, 1 ./ ( Sh0 + diag( sseh ) / 2 ) ) );
                
                Ah = sqrt( Sigh ); % we need this to compute DICs
            else
                % sample Ah, gammah
                [h0, Ah, gammah, S] = SVRSOE(erry, h0, Ah, gammah, h00, ah0, iVh00, iVah0, sharedvol );
                h = repmat( h0, 1, T ) + Ah * gammah;
                ibigSig = spdiags( exp( -h( : ) ), 0, speye( n * T ) );
            end
            
            if forceid
                % transform back to identified parameter space
                [Ux, Dx] = svd( Ax, 0 );
                Cx = Ax \ ( Ux * Dx * diag( sign( Ux( 1, : ) ) ) );
                Ax = Ax * Cx; gammax = Cx' * gammax;
                if sharedvol == 1
                    [Uh, Dh] = svd( Ah, 0 );
                    Ch = Ah \ ( Uh * Dh * diag( sign( Uh( 1, : ) ) ) );
                    Ah = Ah * Ch; gammah = Ch' * gammah;
                end
            end
        else
            % implement shared erros with standard log-volatility
            % specification for h_t
            
            % sample gammax ( = gammah )
            % propose a draw gammac using independence MH
            diffy = y - bigX * theta0;
            XA = hugeX * kron( speye( T ), Ax );
            erry = diffy - XA * gammax( : );
            
            if rx > 0
                H = spdiags( -ones( T * rx, 1 ), -rx, speye( T * rx ) );
                [gam_hat, iVgam] = gam_mode( diffy, XA, Ah, h0, H' * H );
            
                if armh
                    [gammax, erry, accept_gam] = gam_armh( gammax, erry, gam_hat, iVgam, ...
                                                           h0, Ah, diffy, XA, H, T, rx, ...
                                                           nu, ARMHadj, maxARtries, mvtprop );
                else
                    [gammax, erry, accept_gam] = gam_mh( gammax, erry, gam_hat, iVgam, ...
                                                         h0, Ah, h, diffy, XA, H, ...
                                                         T, rx, nu, mvtprop );
                end
            end
            
            % sample h0, Ah
            [h0, Ah, gammah, S] = SVRSOE(erry, h0, Ah, gammax, h00, ah0, iVh00, iVah0, sharedvol );
            h = repmat( h0, 1, T ) + Ah * gammah;
            ibigSig = spdiags( exp( -h( : ) ), 0, speye( n * T ) );
            
            if forceid
                % transform back to identified parameter space
                [U, D] = svd( [Ax; Ah], 0 );
                C = [Ax; Ah] \ ( U * D * diag( sign( U( 1, : ) ) ) );
                Ax = Ax * C; gammax = C' * gammax;
                Ah = Ah * C; gammah = gammax;
            end
        end
    else
        % alternative shared volatility specification not yet implemented
    end
  
    % save draws
    if isim > burnin && mod( isim - burnin, simstep ) == 0
        isave = ( isim - burnin ) / simstep;
        s_theta0( :, isave ) = theta0;
        s_iVth00( :, isave ) = iVth00;
        s_h0( :, isave ) = h0;
        s_Ax( :, 1:rx, isave ) = Ax;
        s_Ah( :, 1:rh, isave ) = Ah;
        s_gammax( 1:rx, :, isave ) = gammax;
        s_gammah( 1:rh, :, isave ) = gammah;
        s_theta( :, :, isave ) = repmat( theta0, 1, T ) + Ax * gammax;
        s_h( :, :, isave ) = h;
        s_sigh( :, isave ) = diag( Sigh );
        s_S( :, isave ) = S;
        s_rx( isave ) = rx;
        s_rh( isave ) = rh;
        s_accept_gam( isave ) = accept_gam;
    end

    % show progress
    txtlen = showprog( isim, burnin + nsims, progstep, txtlen );
end
fprintf( '\nSampling completed after %5.3f minutes.\n', toc / 60 );

[s_Axid, s_gammaxid, s_Ahid, s_gammahid] = trans2id( s_Ax, s_gammax, s_Ah, s_gammah, sharedvol );
mtheta0 = mean( s_theta0, 2 );
mAx = mean( s_Axid, 3 );
mh0 = mean( s_h0, 2 );
mAh = mean( s_Ahid, 3 );
rsoe_DIC;

end
