Sdiff = repmat( dscale', 1, n ) ./ repmat( dscale, n, 1 );
if vecm_form
    R0 = [eye( n ) zeros( n, n * p )];
    R1 = spdiags( [ones( n * ( p + 1 ), 1 ) [ones( n, 1 ); -ones( n * p, 1 )]], ...
               [-n 0], n * ( p + 1 ), n * ( p + 1 ) );
    Slev = repmat( dscale', 1, n ) ./ repmat( dscale_lev, n, 1 );
end

s_A = zeros( n, n, T, svsims );
if vecm_form
    s_B = zeros( n, n * ( p + 1 ), T, svsims );
else
    s_B = zeros( n, n * p, T, svsims );
end
s_c = zeros( n, 1, T, svsims );
for isim = 1:svsims
    for t = 1:T
        sig_t = exp( s_h( :, t, isim ) / 2 );
        Theta_t = zeros( n, n + k );
        Theta_t( freeth ) = s_theta( vec2mat, t, isim );
        
        % reverse the scale and centering
        sig_t = dscale' .* sig_t;
        iA_t = Sdiff .* Theta_t( :, 1:n ) + eye( n );
        if vecm_form
            Pi_t = Slev .* Theta_t( :, n + 2:2 * n + 1 );
            G_t = repmat( Sdiff, 1, p ) .* Theta_t( :, 2 * n + 2:end );
%             d_t = dscale' .* Theta_t( :, n + 1 ) - Pi_t * ( dscale_lev .* dcenter_lev )' ...
%                 + [iA_t -G_t] * repmat( dcenter', p + 1, 1 );
            d_t = dscale' .* Theta_t( :, n + 1 ) - Pi_t * dcenter_lev' ...
                + [iA_t -G_t] * repmat( dcenter', p + 1, 1 );
        else
            G_t = repmat( Sdiff, 1, p ) .* Theta_t( :, n + 2:end );
            d_t = dscale' .* Theta_t( :, n + 1 ) + [iA_t -G_t] * repmat( dcenter', p + 1, 1 );
        end
        
        % recover SVAR parameters
        if vecm_form
            B_t = iA_t \ [Pi_t G_t] * R1 + R0;
        else
            B_t = iA_t \ G_t;
        end
        c_t = iA_t \ d_t;
        A_t = iA_t \ diag( sig_t );
        
        % save draws of the structural parameters
        s_A( :, :, t, isim ) = A_t;
        s_B( :, :, t, isim ) = B_t;
        s_c( :, 1, t, isim ) = c_t;
        
        % test
%         err1( :, t ) = exp( -s_h( :, t, isim ) / 2 ) .* ( Ydata( p + t, : )' - Theta_t * [-Ydata( p + t, : )'; 1; Ylevel( t, : )'; Ydata( p + t - 1, : )'] );
%         err2( :, t ) = sig_t .\ ( Ydiff2( p + t, : )' - ( iA_t - eye( n ) ) * ( -Ydiff2( p + t, : )' ) - d_t - Pi_t * Ylevel2( t, : )' - G_t * Ydiff2( p + t - 1, : )' );
%         err2( :, t ) = sig_t .\ ( iA_t * Ydiff2( p + t, : )' - d_t - Pi_t * Ylevel2( t, : )' - G_t * Ydiff2( p + t - 1, : )' );
%         err3( :, t ) = A_t \ ( Ydata0( 1 + p + t, : )' - c_t - B_t * [Ydata0( p + t, : )'; Ydata0( p + t - 1, : )'] );
%         err4( :, t ) = sig_t .\ ( iA_t * Ydata0( 1 + p + t, : )' - iA_t * Ydata0( p + t, : )' - d_t - [Pi_t G_t] * R1 * [Ydata0( p + t, : )'; Ydata0( p + t - 1, : )'] );
    end
%     err0 = exp( -s_h( :, :, isim ) / 2 ) .* reshape( y - hugeX * vec( s_theta( :, :, isim ) ), n, T );
    
    for t = 1:T
        % do the Barsky & Sims identification procedure to adjust A_t
        extraBs = max( t + 80 - T, 0 ); % for t > T - 80, we need coefficients beyond period T
        IR = ir_tvpvar( cat( 3, s_B( :, :, t:end, isim ), ...
                        repmat( s_B( :, :, end, isim ), [1 1 extraBs] ) ), ...
                        s_A( :, :, t, isim ), 80, 1, 1, 2:n, false );

        Z = squeeze( IR );
        [Vz, Ez] = eig( Z * Z' );
        [~, Zidx] = sort( diag( Ez ), 'descend' );
        Dz = Vz( :, Zidx );

        % normalize the sign---max IR of TFP to news is positive
        [~, maxIR] = max( abs( Z' * Dz( :, 1 ) ) );
        Dz = sign( squeeze( IR( :, :, maxIR ) ) * Dz( :, 1 ) ) * Dz;

        % rotate the VMA matrices
        s_A( :, :, t, isim ) = s_A( :, :, t, isim ) * blkdiag( 1, Dz );
    end
end


%         d_t = dscale' .* ( dcenter' - Theta_t * [-dcenter'; -1; ...
%                            dcenter_lev'; repmat( dcenter', p, 1 )] );
