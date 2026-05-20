function plot_dataset(Y, Ydates, Names)
% PLOT_DATASET  One panel per variable, last (partial) row centered.
%
% All panels rendered at the same size. Small (N=3) uses a 2x4 underlying
% grid where each panel spans 2 cols, so var 3 lands centered at cols 2-3
% of row 2 with the same dimensions as vars 1 and 2.

N = size(Y, 2);

if N == 3
    figure('Name','3 variables','Position',[100 100 900 600]);
    tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile([1 2]);                          % var 1: row 1, cols 1-2
    plot(Ydates, Y(:,1), 'r', 'LineWidth', 1.2);
    title(Names{1}, 'Interpreter','none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
    nexttile([1 2]);                          % var 2: row 1, cols 3-4
    plot(Ydates, Y(:,2), 'r', 'LineWidth', 1.2);
    title(Names{2}, 'Interpreter','none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
    nexttile(6, [1 2]);                       % var 3: row 2, cols 2-3 (centered)
    plot(Ydates, Y(:,3), 'r', 'LineWidth', 1.2);
    title(Names{3}, 'Interpreter','none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
    return;
end

switch N
    case 7,  cols = 3;     % 3x3, last row has 1 var centered (tile 8)
    case 22, cols = 6;     % 4x6, last row has 4 vars centered (tiles 20-23)
    otherwise
        cols = ceil(sqrt(N));
end
rows = ceil(N / cols);

full_rows  = floor(N / cols);
last_row_n = N - full_rows * cols;

if N <= 7
    fig_pos = [100 100 1100 750];
else
    fig_pos = [50 50 1500 850];
end
figure('Name', sprintf('%d variables', N), 'Position', fig_pos);
tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');

fs = max(7, 11 - floor(N/4));

for j = 1:(full_rows * cols)
    nexttile;
    plot(Ydates, Y(:,j), 'r', 'LineWidth', 1);
    title(Names{j}, 'FontSize', fs, 'Interpreter', 'none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
end

if last_row_n > 0
    offset     = floor((cols - last_row_n) / 2);
    start_tile = full_rows * cols + offset + 1;
    for k = 1:last_row_n
        nexttile(start_tile + k - 1);
        plot(Ydates, Y(:, full_rows*cols + k), 'r', 'LineWidth', 1);
        title(Names{full_rows*cols + k}, 'FontSize', fs, ...
              'Interpreter', 'none');
        xtickangle(45);
        try recessionplot; end %#ok<TRYNC>
    end
end
end
