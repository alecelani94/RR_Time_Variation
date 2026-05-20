H = 40;
nmax = 8;
topmarg = 0.17;
vcellpad = 0.08;
height = 0.32;
leftmarg = 0.08;
hcellpad = 0.03;
width = ( 1 - leftmarg - ( nmax - 0.5 ) * hcellpad ) / nmax;
shocks = { {'Non-'; 'News'}, 'News' };
series = { 'Log TFP', 'FED Funds rate', 'Inflation', {'Log hours'; 'per capita'}, {'Log real GDP'; 'per capita'}, {'Log real'; 'consumption'; 'per capita'}, {'Log real'; 'investment'; 'per capita'}, 'Spread' };
set( 0, 'DefaultAxesPosition', [0.1300 0.1100 0.7750 0.8150] );
figure( 'PaperType', 'a4', 'PaperPositionMode', 'manual', ...
        'PaperOrientation', 'landscape', 'PaperUnits', 'normalized', ...
        'PaperPosition', [0 0 1 1], 'Resize', 'off', ...
        'Units', 'centimeters', 'Position', [0.0 0.0 29.7 11] );

ymin = zeros( nmax, 1 );
ymax = zeros( nmax, 1 );
h = zeros( 2, nmax );
for j = 1:2
    for i = 1:nmax
        bounds = squeeze( prctile( s_IR( i, j, 1 + H, :, valid ), [16 84], 5 ) );
        ymin( i ) = min( [ymin( i ); bounds( :, 1 )] );
        ymax( i ) = max( [ymax( i ); bounds( :, 2 )] );
        
        h( j, i ) = subplot( 2, nmax, ( j - 1 ) * nmax + i );
        box on; hold on;
        plot( tper, squeeze( mean( s_IR( i, j, 1 + H, :, valid ), 5 ) ), 'k-', 'LineWidth', 2 );
        plot( tper, bounds, 'r-', 'LineWidth', 2 );
        plot( tper, zeros( length( tper ), 1 ), 'b:', 'LineWidth', 2 );
        
        % adjust display properties
        if j == 1
            % first row, so create titles
            title( series{i}, ...
                'FontSize', 12, 'FontWeight', 'bold' );
        elseif j == 2
            % last row, so create x-axis labels
            xlabel( {'Time'} );
        end
        
        if i == 1
            % first column, so create y-axis labels
            hy = ylabel( shocks{j}, 'Rotation', 0 );
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
for j = 1:2
    for i = 1:nmax
    set( h( j, i ), 'Units', 'Normalized', ...
                    'Position', [leftmarg + (width + hcellpad ) * ( i - 1 ) ...
                    1 - ( topmarg + vcellpad * ( j - 1 ) + height * j ) width height], ...
                    'ylim', [ymin( i ) - 1e-5 ymax( i ) + 1e-5] );
    end
end
print( sprintf( 'irfsH40_n%0.2u_rx%0.2u_rh%0.2u_sharedvol%0.1u', n, rx, rh, sharedvol ), '-depsc', '-painters' );