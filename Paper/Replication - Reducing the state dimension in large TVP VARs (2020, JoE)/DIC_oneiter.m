function [Dy, ln_prior] = DIC_oneiter( theta0, Ax, h0, Ah, gxstar, y, bigX, hugeX, th0_prior, q0, c0, th00, iVth00_0, ax0, iVax0, ah0, iVah0, h00, iVh00, sh0, Sh0, Hh, Hx, initIStries, maxIStries, sharedvol )

[n, rh] = size( Ah );
[~, rx] = size( Ax );
T = length( y ) / n;

if sharedvol < 3
    diffy = y - bigX * theta0;
    XA = hugeX * kron( speye( T ), Ax );
    if rh > 0
        if sharedvol < 2
            % Do numerical integration over 'gammah' only, and use EM to design a proposal
            [gamh_hat, iVgamh] = gamh_mode( diffy, XA, Ah, h0, Hh, Hx, sharedvol );

            % inline function to evaluate log p( h ) * p( y | h )
            f = @( x ) -0.5 * ( rh * T * log( 2 * pi ) + sum( ( Hh * x ) .^ 2 ) ) ...
                  - 0.5 * ( rx * T * log( 2 * pi ) + sum( ( Hx * gxstar ) .^ 2 ) ) ...
                  - 0.5 * ( n * T * log( 2 * pi ) + sum( vec( repmat( h0, 1, T ) ...
                  + Ah * reshape( x, rh, T ) ) ...
                  + exp( -vec( repmat( h0, 1, T ) + Ah * reshape( x, rh, T ) ) ) ...
                  .* ( diffy - XA * gxstar ) .^ 2 ) ) ...
                  - lnpr_gamx_given_gamh( gxstar, diffy, XA, ...
                  repmat( h0, 1, T ) + Ah * reshape( x, rh, T ), Hx' * Hx );

            % inline function to evaluate the log proposal
            q = @( x ) -0.5 * ( rh * T * log( 2 * pi ) - 2 * sum( log( diag( chol( iVgamh ) ) ) ) ...
                   + ( x - gamh_hat )' * iVgamh * ( x - gamh_hat ) );

            % Now just do basic importance sampling
            ln_likei = zeros( maxIStries, 1 );
            gamma = repmat( gamh_hat, 1, initIStries ) ...
                   + chol( iVgamh ) \ randn( rh * T, initIStries );
               
            for jsim = 1:initIStries
                ln_likei( jsim ) = f( gamma( :, jsim ) ) - q( gamma( :, jsim ) );
            end
            Vz = llestvar( ln_likei( 1:initIStries ), 100 );

    	    % if countIS < maxIStries, we're trying to select the number
            % of particles dynamically using the procedure in Pitt, et. al. (2012) 
            countIS = initIStries;
            while Vz > 0.85 && countIS < maxIStries % using the threshold suggested in Pitt, et. al. (2012)
                countIS = countIS + 1;
                gamma = gamh_hat + chol( iVgamh ) \ randn( rh * T, 1 );
                ln_likei( countIS ) = f( gamma ) - q( gamma );
                Vz = llestvar( ln_likei( 1:countIS ), 100 );
            end

    	    mln_likeij = mean( ln_likei( 1:countIS ) );
            llest = log( mean( exp( ln_likei( 1:countIS ) - mln_likeij ) ) ) + mln_likeij;
        else
            % Integrate 'gammax, gammah' jointly using a Newton-Raphson based proposal
            [gam_hat, iVgam] = gam_mode( diffy, XA, Ah, h0, Hx' * Hx );
            llest = lnlike_peis( diffy, XA, Ah, h0, Hx, gam_hat, iVgam, maxIStries );
        end
        Dy = -2 * llest;
    elseif rx > 0
        llest = - 0.5 * ( rx * T * log( 2 * pi ) + sum( ( Hx * gxstar ) .^ 2 ) ) ...
                - 0.5 * ( n * T * log( 2 * pi ) + T * sum( h0 ) ...
                + sum( exp( -repmat( h0, T, 1 ) ) .* ( diffy - XA * gxstar ) .^ 2 ) ) ...
                - lnpr_gamx_given_gamh( gxstar, diffy, XA, repmat( h0, 1, T ), Hx' * Hx );
        Dy = -2 * llest;
    else
        h = repmat( h0, T, 1 );
        Dy = n * T * log( 2 * pi ) + sum( h + diffy .^ 2 .* exp( -h ) );
    end
else
	% not yet implemented
end

% evaluate log prior

if th0_prior == 1
    % lasso prior
    ln_pr_theta0 = 0;
    % to be completed
else
    % SSVS or conventional normal
    q00 = q0 * ( 1 - ( th0_prior == 0 ) );
    ln_pr_theta0 = sum( log( q00 * exp( -0.5 * iVth00_0 .* ( theta0 - th00 ) .^ 2 ) ...
        + ( 1 - q00 ) / sqrt( c0 ) * exp( -0.5 * iVth00_0 .* ( theta0 - th00 ) .^ 2 / c0 ) ) );
end

% compute the ln prior assuming elememnts in Ax and Ah are iid and centered on zero
ln_pr_Ax = 0; ln_pr_Ah = 0;
if rx > 0
    ln_pr_Ax = -0.5 * trace( Ax' * Ax ) * iVax0( 1 );
end

if sharedvol > 0
    if rh > 0
        ln_pr_Ah = -0.5 * trace( Ah' * Ah ) * iVah0( 1 );
    end
else
    % this will actually be the inverse-gamma prior
    ln_pr_Ah = -2 * sum( ( sh0 + 1 ) .* log( diag( Ah ) ) - Sh0 ./ diag( Ah ) .^ 2 );
end

ln_prior = ln_pr_theta0 + ln_pr_Ax + ln_pr_Ah - 0.5 * sum( ( h0 - h00 ) .^ 2 .* iVh00 );