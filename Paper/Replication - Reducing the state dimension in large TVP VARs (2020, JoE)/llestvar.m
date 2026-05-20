function out = llestvar( draws, nboot )
% Compute the Monte Carlo variance of a log likelihood estimator
% using a bootstrap.
%
% draws: MC draws to bootstrap
% nboot: number of bootsrapped sampes

llest = @( x ) log( mean( exp( x - mean( x ) ) ) ) + mean( x );
out = var( bootstrp( nboot, llest, draws ) );
