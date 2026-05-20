% compute DIC for models with reduced errors and fixed rx, rh
if ~exist( 'doparallel', 'var' )
    doparallel = true;
end
initIStries = 50;
maxIStries = initIStries; % can be different if using an adaptive algorithm
progstep = 100;
rx = rmaxx;
rh = rmaxh;
if rh > 0
    Hh = spdiags( -ones( T * rh, 1 ), -rh, speye( T * rh ) );
else
    Hh = [];
end
if rx > 0
    Hx = spdiags( -ones( T * rx, 1 ), -rx, speye( T * rx ) );
else
    Hx = [];
end
gxstar = vec( mean( s_gammax, 3 ) );

% we need to augment the stored draws matrices for parfor loop slicing to
% work properly (e.g. to avoid expensive data transfers at each iter.)
ss_theta0 = [s_theta0 mtheta0];
ss_Ax = cat( 3, s_Ax, mAx );
ss_h0 = [s_h0 mh0];
ss_Ah = cat( 3, s_Ah, mAh );

s_Dy = zeros( svsims + 1, 1 );
s_ln_prior = zeros( svsims + 1, 1 );

disp( ['Computing DIC for sharedvol=' num2str( sharedvol ) ' model...'] );

% the following is needed ONLY on Windows
% to display progress properly; on Linux
% it has no effect (but should be left defined)
txtlen = 0;
    
if doparallel
    clear txtlen;
    if exist( 'parpool' ) && isempty( gcp )
        parpool; % start a new parallel pool if not already running
    elseif exist( 'matlabpool' ) && matlabpool( 'size' ) == 0
        matlabpool; % start a new parallel pool if not already running
    end
%     dpath = javaclasspath;
%     if isempty( dpath ) || ~strcmpi( dpath, [pwd '\java'] )
%         pctRunOnAll javaaddpath java;
%     end
    ppm = ParforProgMon( 'DIC computation: ', svsims + 1, floor( ( svsims + 1 ) / progstep ), 300, 80 );
end

% initialize the random number generator
% rand( 'state', sum( 100 * clock ) );
% randn( 'state', sum( 200 * clock ) );
rng( 'shuffle' ); rngstate = rng; % a new state
% load rngstate-001; rng( rngstate ); % a previously saved state

tic; % start the clock
if doparallel
    parfor isim = 1:svsims + 1 % parallel IS
        theta0 = ss_theta0( :, isim );
        Ax = ss_Ax( :, :, isim );
        h0 = ss_h0( :, isim );
        Ah = ss_Ah( :, :, isim );
    
        [Dy, ln_prior] = DIC_oneiter( theta0, Ax, h0, Ah, gxstar, ...
                                      y, bigX, hugeX, th0_prior, ...
                                      q0, c0, th00, iVth00_0, ax0, iVax0, ...
                                      ah0, iVah0, h00, iVh00, sh0, Sh0, ...
                                      Hh, Hx, initIStries, maxIStries, sharedvol );
        s_Dy( isim ) = Dy;
        s_ln_prior( isim ) = ln_prior;
    
        % show progress
        if mod( isim, floor( ( svsims + 1 ) / progstep ) ) == 0
            ppm.increment();
        end
    end
else
    for isim = 1:svsims + 1 % sequential IS
        theta0 = ss_theta0( :, isim );
        Ax = ss_Ax( :, :, isim );
        h0 = ss_h0( :, isim );
        Ah = ss_Ah( :, :, isim );
    
        [Dy, ln_prior] = DIC_oneiter( theta0, Ax, h0, Ah, gxstar, ...
                                      y, bigX, hugeX, th0_prior, ...
                                      q0, c0, th00, iVth00_0, ax0, iVax0, ...
                                      ah0, iVah0, h00, iVh00, sh0, Sh0, ...
                                      Hh, Hx, initIStries, maxIStries, sharedvol );
        s_Dy( isim ) = Dy;
        s_ln_prior( isim ) = ln_prior;
    
        % show progress
        txtlen = showprog( isim, svsims + 1, progstep, txtlen );
    end
end

% compute the DIC by evaluating D(.) at a posterior estimate (e.g. mean)
Dbar1 = s_Dy( end );
p_D1 = mean( s_Dy( 1:svsims ) ) - Dbar1;
DIC1 = mean( s_Dy( 1:svsims ) ) + p_D1;

% compute the DIC by evaluating D(.) at the posterior mode
[~, mapid] = max( s_ln_prior - s_Dy / 2 );
Dbar2 = s_Dy( mapid );
p_D2 = mean( s_Dy( 1:svsims ) ) - Dbar2;
DIC2 = mean( s_Dy( 1:svsims ) ) + p_D2;

% compute the DIC by evaluating D(.) at the estimated post pred density
fDIC3 = @( x ) mean( x ) + 2 * log( mean( exp( - ( x - mean( x ) ) / 2 ) ) );
DIC3 = fDIC3( s_Dy( 1:svsims ) );
p_D3 = DIC3 - mean( s_Dy( 1:svsims ) );
seDIC3 = std( bootstrp( 10000, fDIC3, s_Dy( 1:svsims ) ) );


if doparallel
    ppm.delete();
end
fprintf( '\nDIC computation completed after %5.3f minutes.\n', toc / 60 );

% [mtheta0, mAx, mh0, mAh] = quickmode( y, X, bigX, hugeX, Xones, th00, ax0, h00, ah0, iVax0, iVh00, iVah0, sh0, Sh0, s_theta0, s_Ax, s_h0, s_Ah, s_sigh, s_gammax, s_gammah, s_h, s_iVth00, s_S, k, sharedvol );
% mtheta0 = margmode( s_theta0 );
% mAx = margmode( s_Ax );
% mh0 = margmode( s_h0 );
% mAh = margmode( s_Ah );
% mAh = diag( margmode( sqrt( s_sigh ) ) );
% mtheta0 = mean( s_mtheta0, 2 );
% mAx = mean( s_mAx, 3 );
% mh0 = mean( s_mh0, 2 );
% mAh = mean( s_mAh, 3 );
% mtheta0 = mean( s_theta0, 2 );
% mAx = mean( s_Ax, 3 );
% mh0 = mean( s_h0, 2 );
% mAh = mean( s_Ah, 3 );

