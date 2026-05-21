function plot_dataset(Y, Ydates, Names)
% PLOT_DATASET  One panel per variable, last (partial) row centered.
%
% All panels rendered at the same size. For cases where the natural
% rows x cols grid would leave an odd-count last row, we use a
% finer-resolution tiledlayout with each panel spanning multiple cols
% so the last row sits visually centered without resizing panels.

N = size(Y, 2);

% ---- Small datasets with bespoke centered layouts ----
if N == 3
    figure('Name','3 variables','Position',[100 100 900 600]);
    tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile([1 2]);                          % var 1: row 1, cols 1-2
    plot_one(Ydates, Y(:,1), Names{1});
    nexttile([1 2]);                          % var 2: row 1, cols 3-4
    plot_one(Ydates, Y(:,2), Names{2});
    nexttile(6, [1 2]);                       % var 3: row 2, cols 2-3 (centered)
    plot_one(Ydates, Y(:,3), Names{3});
    return;
end

if N == 4
    figure('Name','4 variables','Position',[100 100 1000 700]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for j = 1:4
        nexttile;
        plot_one(Ydates, Y(:,j), Names{j});
    end
    return;
end

if N == 5
    % 2x6 underlying grid; each panel spans 2 cols. Row 1: vars 1-3 in
    % tiles 1-2 / 3-4 / 5-6. Row 2: vars 4-5 in tiles 8-9 / 10-11 (centered,
    % skipping cols 1 and 6).
    figure('Name','5 variables','Position',[100 100 1200 700]);
    tiledlayout(2, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile([1 2]);  plot_one(Ydates, Y(:,1), Names{1});
    nexttile([1 2]);  plot_one(Ydates, Y(:,2), Names{2});
    nexttile([1 2]);  plot_one(Ydates, Y(:,3), Names{3});
    nexttile(8,  [1 2]);  plot_one(Ydates, Y(:,4), Names{4});
    nexttile(10, [1 2]);  plot_one(Ydates, Y(:,5), Names{5});
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

function plot_one(Ydates, y, name)
    plot(Ydates, y, 'r', 'LineWidth', 1.2);
    title(name, 'Interpreter','none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
end
