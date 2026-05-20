% genDICs.m
irun = 55;  % set the identifier of this run
n = 8; % size of TVP-VARs being estimated
rtotal = 3; % total number of err sources (rmaxx + rmaxh)

% the grid for parsing different combinations of rx, rh
if rtotal == 0
    rj2rmax = 0;
elseif rtotal == 1
    rj2rmax = [0 1];
elseif rtotal == 3
    rj2rmax = [0 1 2 3];
elseif rtotal == 4
    rj2rmax = [0 1 2 3 4];
elseif rtotal == 5
    rj2rmax = [0 1 2 3 4 5];
elseif rtotal == 6
    rj2rmax = [0 1 2 4 5 6];
elseif rtotal == 7
	rj2rmax = [0 1 3 4 6 7];
elseif rtotal == 8
    rj2rmax = [0 2 3 4 5 6 8];
elseif rtotal == 10
    if n >= 10
    	rj2rmax = [0 2 4 5 6 8 10];
    elseif n >= 8
        rj2rmax = [2 4 5 6 8 10];
    else
        rj2rmax = [];
    end
elseif rtotal == 12
    if n >= 12
    	rj2rmax = [0 4:8 12];
    elseif n >= 8
        rj2rmax = [4:8 12];
    else
        rj2rmax = [];
    end
else
    rj2rmax = nan;
end

if exist( 'parpool' ) && isempty( gcp )
	parpool; % start a new parallel pool if not already running
elseif exist( 'matlabpool' ) && matlabpool( 'size' ) == 0
	matlabpool; % start a new parallel pool if not already running
end

sharedvol = 1;  % reduced errors in coeff and independently volatilities
parfor rj = 1:length( rj2rmax )
    % run the script for rmaxx + rmaxh = rtotal
    rmaxx = rj2rmax( rj );
    rmaxh = rtotal - rmaxx;

    [s_Dy, s_ln_prior, DIC1, p_D1, DIC2, p_D2, DIC3, p_D3, seDIC3] = rsoe_in_loop( rmaxx, rmaxh, sharedvol, false );

    fname = ['DIC_n' num2str( n, '%02u' ) '_rx' num2str( rmaxx ) '_rh' num2str( rmaxh ) '_share' num2str( sharedvol ) '_run' num2str( irun, '%02u' )];
    mfile = matfile( fname, 'writable', true );
    mfile.rmaxx = rmaxx;
    mfile.rmaxh = rmaxh;
    mfile.sharedvol = sharedvol;
    mfile.s_Dy = s_Dy;
    mfile.s_ln_prior = s_ln_prior;
    mfile.DIC1 = DIC1; mfile.p_D1 = p_D1;
    mfile.DIC2 = DIC2; mfile.p_D2 = p_D2;
    mfile.DIC3 = DIC3; mfile.p_D3 = p_D3; mfile.seDIC3 = seDIC3;
end

sharedvol = 2;  % error sharing between volatilies and coeff
rmaxx = rtotal; rmaxh = 0;
[s_Dy, s_ln_prior, DIC1, p_D1, DIC2, p_D2, DIC3, p_D3, seDIC3] = rsoe_in_loop( rmaxx, rmaxh, sharedvol, true );
fname = ['DIC_n' num2str( n, '%02u' ) '_rx' num2str( rtotal ) '_rh' num2str( rtotal ) '_share' num2str( sharedvol ) '_run' num2str( irun, '%02u' )];
save( fname, 'rmaxx', 'rmaxh', 'sharedvol', 's_Dy', 's_ln_prior', 'DIC1', 'p_D1', 'DIC2', 'p_D2', 'DIC3', 'p_D3', 'seDIC3' );