H = 80;
topmarg = 0.1;
vcellpad = 0.04;
height = 0.245;
leftmarg = 0.1;
hcellpad = 0.03;
width = ( 1 - leftmarg - ( n - 0.5 ) * hcellpad ) / n;
shocks = { {'Non-news'; 'shock'}, {'News'; 'shock'}, {'Noise'; 'shock'} };
series = { 'Log TFP', 'FED Funds rate', 'Inflation', {'Log hours'; 'per capita'}, {'Log real GDP'; 'per capita'}, {'Log real'; 'consumption'; 'per capita'}, {'Log real'; 'investment'; 'per capita'}, 'Spread' };
set( 0, 'DefaultAxesPosition', [0.1300 0.1100 0.7750 0.8150] );
figure( 'PaperType', 'a4', 'PaperPositionMode', 'manual', ...
        'PaperOrientation', 'landscape', 'PaperUnits', 'normalized', ...
        'PaperPosition', [0 0 1 1], 'Resize', 'off', ...
        'Units', 'centimeters', 'Position', [0.0 0.0 29.7 21] );

h = zeros( 3, n );
for j = 1:3
    for i = 1:n
        h( j, i ) = subplot( 3, n, ( j - 1 ) * n + i );
        box on; hold on;
        plot( 0:irf_hrz, median( FEVD( i, 1:irf_hrz + 1, j, valid ), 4 ), 'k-', 'LineWidth', 2 );
        plot( 0:irf_hrz, squeeze( prctile( FEVD( i, 1:irf_hrz + 1, j, valid ), [16 84], 4 ) ), 'r-', 'LineWidth', 2 );
        
        % adjust display properties
        ylim( [-0.000001, 1.000001] );
        if j == 1
            % first row, so create titles
            title( series{i}, ...
                'FontSize', 12, 'FontWeight', 'bold' );
        elseif j == 3
            % last row, so create x-axis labels
            xlabel( {'Horizon (quarters'; 'after shock)'} );
        end
        
        if i == 1
            % first column, so create y-axis labels
            hy = ylabel( shocks{j}, 'Rotation', 0 );
            set( hy, 'Units', 'Normalized', ...
                'VerticalAlignment', 'middle', ...
                'HorizontalAlignment', 'center', ...
                'Position', [-0.7, 0.5, 0], ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'Color', 'b' );
        end
        hold off;
    end
end
drawnow;

% re-align and re-scale the subplots
for j = 1:3
    for i = 1:n
    set( h( j, i ), 'Units', 'Normalized', ...
                    'Position', [leftmarg + (width + hcellpad ) * ( i - 1 ) ...
                    1 - ( topmarg + vcellpad * ( j - 1 ) + height * j ) ...
                    width height] );
    end
end