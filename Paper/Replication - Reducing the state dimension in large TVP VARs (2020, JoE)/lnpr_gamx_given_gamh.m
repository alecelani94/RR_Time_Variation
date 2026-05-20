function out = lnpr_gamx_given_gamh( gamx, diffy, XA, h, HHx )
[n, T] = size( h );
rx = length( gamx ) / T;

ibigSig = spdiags( exp( -h( : ) ), 0, n * T, n * T );
XAibigSig = XA' * ibigSig;
iVgamx = HHx + XAibigSig * XA;
gamx_hat = iVgamx \ ( XAibigSig * diffy );
out = -0.5 * ( rx * T * log( 2 * pi ) - 2 * sum( log( diag( chol( iVgamx ) ) ) ) ...
      + ( gamx - gamx_hat )' * iVgamx * ( gamx - gamx_hat ) );