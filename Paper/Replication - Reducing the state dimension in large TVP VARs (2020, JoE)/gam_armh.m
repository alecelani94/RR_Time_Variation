function [gammax, erry, accept_gam] = gam_armh( gammax, erry, gam_hat, iVgam, h0, Ah, diffy, XA, H, T, rx, nu, ARMHadj, maxARtries, mvtprop )

r = rx; % (should be equal to rh)

% check that iVgam is positive definite
[chol_iVgam, p] = chol( iVgam );
if p
    diVgam = min( max( abs( diag( iVgam ) ), 1 ), 1e3 );
    iVgam = diag( diVgam );
    chol_iVgam = diag( sqrt( diVgam ) );
end

f = @( x ) -0.5 * sum( ( H * x ) .^ 2 ) ...
          - 0.5 * sum( vec( repmat( h0, 1, T ) + Ah * reshape( x, r, T ) ) ...
          + exp( -vec( repmat( h0, 1, T ) + Ah * reshape( x, r, T ) ) ) ...
          .* ( diffy - XA * x ) .^ 2 );

if mvtprop
    q = @( x ) -( n + nu ) / 2 * log( nu + ( x - gam_hat )' * iVgam * ( x - gam_hat ) );
else
    q = @( x ) -0.5 * ( x - gam_hat )' * iVgam * ( x - gam_hat );
end

% compute the AR constant using Chib and Jeliazkov (2009) rule of thumb
ln_c = log( ARMHadj ) + f( gam_hat ) - q( gam_hat );

% get an initial candidate draw
itau = sqrt( 1 + mvtprop * ( gamrnd( nu / 2, 2 / nu ) - 1 ) );
gammac = reshape( gam_hat + ( itau * chol_iVgam ) \ randn( r * T, 1 ), r, T ) ;
ln_pr_ARc = f( gammac( : ) ) - q( gammac( : ) ) - ln_c;

% repeat until a draw is accepted
countAR = 0;
while log( rand ) >= ln_pr_ARc && countAR < maxARtries
    itau = sqrt( 1 + mvtprop * ( gamrnd( nu / 2, 2 / nu ) - 1 ) );
    gammac = reshape( gam_hat + ( itau * chol_iVgam ) \ randn( r * T, 1 ), r, T ) ;
    ln_pr_ARc = f( gammac( : ) ) - q( gammac( : ) ) - ln_c;
    countAR = countAR + 1;
end

if countAR < maxARtries
    % a draw was accepted, so go to the MH step
    ln_pr_AR = f( gammax( : ) ) - q( gammax( : ) ) - ln_c;
    
    if ln_pr_AR < 0
        ln_pr_accept = 1;
    elseif ln_pr_ARc < 0
        ln_pr_accept = -ln_pr_AR;
    else
        ln_pr_accept = ln_pr_ARc - ln_pr_AR;
    end
    
    accept_gam = log( rand ) < ln_pr_accept;
    if accept_gam
        gammax = gammac;
        erry = diffy - XA * gammac( : );
    end
else
    fprintf( '\nARMH: failed to get a valid candidate draw.\n\n' );
    accept_gam = 0;
end