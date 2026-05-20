shock = 2; % news
irf_hrz = 40;
nmax = 8;
tstep = 35;
topmarg = 0.1;
vcellpad = 0.04;
height = 0.13;
leftmarg = 0.08;
hcellpad = 0.03;
width = ( 1 - leftmarg - ( nmax - 0.5 ) * hcellpad ) / nmax;
s = length( tper( tstep:tstep:end ) );
quarters = { 'Q1', 'Q2', 'Q3', 'Q4' };
series = { 'Log TFP', 'FED Funds rate', 'Inflation', {'Log hours'; 'per capita'}, {'Log real GDP'; 'per capita'}, {'Log real'; 'consumption'; 'per capita'}, {'Log real'; 'investment'; 'per capita'}, 'Spread' };
set( 0, 'DefaultAxesPosition', [0.1300 0.1100 0.7750 0.8150] );
figure( 'PaperType', 'a4', 'PaperPositionMode', 'manual', ...
        'PaperOrientation', 'landscape', 'PaperUnits', 'normalized', ...
        'PaperPosition', [0 0 1 1], 'Resize', 'off', ...
        'Units', 'centimeters', 'Position', [0.0 0.0 29.7 21] );

h = zeros( s, nmax );
for t = 1:s
    for i = 1:nmax
        h( t, i ) = subplot( s, nmax, ( t - 1 ) * nmax + i );
        box on; hold on;
        plot( 0:irf_hrz, squeeze( mean( s_FEVD( i, shock, 1:irf_hrz + 1, t * tstep, valid ), 5 ) ), 'k-', 'LineWidth', 2 );
        plot( 0:irf_hrz, squeeze( prctile( s_FEVD( i, shock, 1:irf_hrz + 1, t * tstep, valid ), [16 84], 5 ) ), 'r-', 'LineWidth', 2 );
        
        % adjust display properties
        ylim( [-0.000001, 1.000001] );
        if t == 1
            % first row, so create titles
            title( series{i}, ...
                'FontSize', 12, 'FontWeight', 'bold' );
        elseif t == s
            % last row, so create x-axis labels
            xlabel( {'Horizon (quarters'; 'after shock)'} );
        end
        
        if i == 1
            % first column, so create y-axis labels
            yr = fix( tper( t * tstep ) );
            hy = ylabel( [num2str( yr ) ...
                quarters( 1 + 4 * ( tper( t * tstep ) - yr ) )], 'Rotation', 0 );
            set( hy, 'Units', 'Normalized', ...
                'VerticalAlignment', 'middle', ...
                'HorizontalAlignment', 'center', ...
                'Position', [-0.5, 0.5, 0], ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'Color', 'b' );
        end
        hold off;
    end
end
drawnow;

% re-align and re-scale the subplots
for t = 1:s
    for i = 1:nmax
    set( h( t, i ), 'Units', 'Normalized', ...
                    'Position', [leftmarg + (width + hcellpad ) * ( i - 1 ) ...
                    1 - ( topmarg + vcellpad * ( t - 1 ) + height * t ) width height] );
    end
end
print( sprintf( 'fevds2news_n%0.2u_rx%0.2u_rh%0.2u_sharedvol%0.1u', n, rx, rh, sharedvol ), '-depsc', '-painters' );