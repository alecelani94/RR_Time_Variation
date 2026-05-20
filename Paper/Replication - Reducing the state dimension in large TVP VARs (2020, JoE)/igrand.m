function out = igrand( theta, chi )

n = size( theta );

if ( n ~= size( chi ) )
	error 'Parameter matrices must be the same size.';
end

chisq1 = randn( n ) .^ 2;

out = theta + 0.5 * theta ./ chi .* ( theta .* chisq1 - sqrt( 4 * theta .* chi .* chisq1 + theta .^ 2 .* chisq1 .^ 2 ) );

l = rand( n ) >= theta ./ ( theta + out );
out( l ) = theta( l ) .^ 2 ./ out( l );
