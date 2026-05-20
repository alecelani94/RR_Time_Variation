function [gammax, erry, accept_gam] = gam_mh( gammax, erry, gam_hat, iVgam, h0, Ah, h, diffy, XA, H, T, rx, nu, mvtprop )

itau2 = 1 + mvtprop * ( gamrnd( nu / 2, 2 / nu ) - 1 );
gammac = reshape( gam_hat + chol( itau2 * iVgam ) \ randn( rx * T, 1 ), rx, T ) ;
            
erryc = diffy - XA * gammac( : );
hc = repmat( h0, 1, T ) + Ah * gammac;
            
% compute the log (unbounded) acceptance ratio
ln_prior_rat = -0.5 * sum( ( H * gammac( : ) ) .^ 2 ) ...
	           +0.5 * sum( ( H * gammax( : ) ) .^ 2 );

ln_like_rat = -0.5 * sum( hc( : ) + exp( -hc( : ) ) .* erryc .^ 2 ) ...
              +0.5 * sum( h( : ) + exp( -h( : ) ) .* erry .^ 2 );

if mvtprop
    % for the multivariate t proposal
    ln_prop_rat = -( n + nu ) / 2 * log( nu + ( gammac( : ) - gam_hat )' * ...
                                      iVgam * ( gammac( : ) - gam_hat ) ) ...
                  +( n + nu ) / 2 * log( nu + ( gammax( : ) - gam_hat )' ...
                                    * iVgam * ( gammax( : ) - gam_hat ) );
else
    % for the multivariate normal proposal
    ln_prop_rat = -0.5 * ( gammac( : ) - gam_hat )' * ...
                 iVgam * ( gammac( : ) - gam_hat ) ...
                  +0.5 * ( gammax( : ) - gam_hat )' * ...
                 iVgam * ( gammax( : ) - gam_hat );
end
            
ln_pr_accept = ln_prior_rat + ln_like_rat - ln_prop_rat;
accept_gam = log( rand ) < ln_pr_accept;
if accept_gam
    gammax = gammac;
    erry = erryc;
end