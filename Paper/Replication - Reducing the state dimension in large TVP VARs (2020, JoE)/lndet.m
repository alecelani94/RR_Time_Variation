function out = lndet( X )

if issparse( X )
    out = 2 * sum( log( diag( chol( X ) ) ) );
else
    out = sum( log( svd( X ) ) );
end