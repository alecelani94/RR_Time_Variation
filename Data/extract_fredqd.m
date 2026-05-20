clear
close all
clc

% =========================================================================
% Extract GLP (2015) datasets from FRED-QD
% Small (3), Medium (7), Large (23) BVAR variable sets
%
% Reads the most recent *-QD.csv in this folder, applies GLP transformations
% (400 * Delta log for growth series, raw for rates/utilization), and saves
% to fredqd_glp.mat as three cell entries.
% =========================================================================

%% ----- Load CSV (auto-pick latest, auto-detect delimiter) ---------------

here = fileparts(mfilename('fullpath'));
csv_files = dir(fullfile(here, '*-QD.csv'));
if isempty(csv_files)
    error('No *-QD.csv files found in %s', here);
end
[~, idx] = sort({csv_files.name});
csv_in = fullfile(here, csv_files(idx(end)).name);
fprintf('Using CSV: %s\n', csv_files(idx(end)).name);

fid = fopen(csv_in, 'r');
first_line = fgetl(fid);
if sum(first_line == ';') > sum(first_line == ',')
    delim = ';';
else
    delim = ',';
end

header = strsplit(first_line, delim);
series = header(2:end);
N = length(series);

fgetl(fid);  % skip factor indicators line
fgetl(fid);  % skip transformation codes line (we apply GLP transforms)

% Preallocate generously; trim at the end.
cap = 2000;
raw_dates  = cell(cap, 1);
raw_values = NaN(cap, N);
row = 0;
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(strtrim(line))
        idx_d = find(line == delim);
        row = row + 1;
        if row > cap
            cap = 2 * cap;
            raw_dates{cap, 1} = '';
            raw_values(cap, N) = NaN;
        end
        raw_dates{row, 1} = line(1:idx_d(1)-1);
        vals = NaN(1, N);
        for j = 1:N
            if j < N
                tok = line(idx_d(j)+1 : idx_d(j+1)-1);
            else
                tok = line(idx_d(j)+1 : end);
            end
            vals(j) = str2double(tok);
        end
        raw_values(row, :) = vals;
    end
end
fclose(fid);
raw_dates  = raw_dates(1:row);
raw_values = raw_values(1:row, :);

%% ----- Parse dates ------------------------------------------------------

T = length(raw_dates);
dates = NaT(T,1);
for t = 1:T
    dates(t) = datetime(raw_dates{t}, 'InputFormat', 'M/d/yyyy');
end

%% ----- Define GLP variable sets -----------------------------------------
% FRED mnemonic, label, transform ('4log' or 'raw')
%   '4log' = 400 * Delta log(x_t)  (annualized quarterly log growth)
%   'raw'  = no transformation (levels)

small_vars = {
    'GDPC1',        'RGDP',             '4log'
    'GDPCTPI',      'PGDP',             '4log'
    'FEDFUNDS',     'FedFunds',         'raw'
};

medium_vars = {
    'GDPC1',        'RGDP',             '4log'
    'GDPCTPI',      'PGDP',             '4log'
    'FEDFUNDS',     'FedFunds',         'raw'
    'PCECC96',      'Cons',             '4log'
    'GPDIC1',       'Inv',              '4log'
    'HOANBS',       'EmpHours',         '4log'
    'COMPRNFB',     'RealCompHour',     '4log'
};

large_vars = {
    'GDPC1',        'RGDP',             '4log'
    'GDPCTPI',      'PGDP',             '4log'
    'FEDFUNDS',     'FedFunds',         'raw'
    'CPIAUCSL',     'CPIALL',           '4log'
    'OILPRICEx',    'ComSpotPrice',     '4log'
    'INDPRO',       'IPtotal',          '4log'
    'PAYEMS',       'Emptotal',         '4log'
    'SRVPRD',       'EmpServices',      '4log'
    'PCECC96',      'Cons',             '4log'
    'GPDIC1',       'Inv',              '4log'
    'PRFIx',        'ResInv',           '4log'
    'PNFIx',        'NonResInv',        '4log'
    'PCECTPI',      'PCED',             '4log'
    'GPDICTPI',     'PGPDI',            '4log'
    'TCU',          'CapacityUtil',     'raw'
    'UMCSENTx',     'ConsExpect',       'raw'
    'HOANBS',       'EmpHours',         '4log'
    'COMPRNFB',     'RealCompHour',     '4log'
    'GS1',          'GS1',              'raw'
    'GS10',         'GS10',             'raw'
    'S&P 500',      'SP500',            '4log'
    'TWEXAFEGSMTHx','ExRate',           '4log'
};

%% ----- Build datasets ---------------------------------------------------
% Output: Y{d}      = T-1 x n double matrix (post-differencing)
%         Ynames{d} = 1 x n cell of variable name strings
%         Ydates     = T-1 x 1 datetime vector (shared across datasets)

datasets = {small_vars, medium_vars, large_vars};
dsnames  = {'Small', 'Medium', 'Large'};
Y      = cell(1, 3);
Ynames = cell(1, 3);

for d = 1:3
    vars = datasets{d};
    nv = size(vars, 1);

    raw = NaN(T, nv);
    for j = 1:nv
        idx = find(strcmp(series, vars{j,1}));
        if isempty(idx)
            error('Series %s not found in FRED-QD.', vars{j,1});
        end
        raw(:, j) = raw_values(:, idx);
    end

    transformed = NaN(T, nv);
    for j = 1:nv
        if strcmp(vars{j,3}, '4log')
            transformed(2:end, j) = 400 * (log(raw(2:end,j)) - log(raw(1:end-1,j)));
        else
            transformed(:, j) = raw(:, j);
        end
    end

    transformed = transformed(2:end, :);  % drop first obs (lost to differencing)

    Y{d}      = transformed;
    Ynames{d} = vars(:,2)';

    fprintf('%s BVAR: %2d variables, %d quarters (%s to %s)\n', ...
        dsnames{d}, nv, size(transformed,1), ...
        string(dates(2)), string(dates(end)));
end

Ydates = dates(2:end);

%% ----- Save (three separate .mat files) ---------------------------------

out_files = {'fredqd_small.mat', 'fredqd_medium.mat', 'fredqd_large.mat'};
for d = 1:3
    out_path = fullfile(here, out_files{d});
    data   = Y{d};
    names  = Ynames{d};
    dates_ = Ydates;
    save(out_path, 'data', 'names', 'dates_');
    fprintf('Saved %-22s  %d x %2d\n', out_files{d}, size(Y{d},1), size(Y{d},2));
end

%% ----- Plots ------------------------------------------------------------

% Small (3 vars) -- 2x2 visual, panels same size via 2x4 underlying grid
figure('Name','Small (3 vars)','Position',[100 100 900 600]);
tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile([1 2]);                  % var 1: row 1, cols 1-2
plot(Ydates, Y{1}(:,1), 'r', 'LineWidth', 1.2);
title(Ynames{1}{1}, 'Interpreter','none');
xtickangle(45);
try recessionplot; end %#ok<TRYNC>
nexttile([1 2]);                  % var 2: row 1, cols 3-4
plot(Ydates, Y{1}(:,2), 'r', 'LineWidth', 1.2);
title(Ynames{1}{2}, 'Interpreter','none');
xtickangle(45);
try recessionplot; end %#ok<TRYNC>
nexttile(6, [1 2]);               % var 3: row 2, cols 2-3 (centered)
plot(Ydates, Y{1}(:,3), 'r', 'LineWidth', 1.2);
title(Ynames{1}{3}, 'Interpreter','none');
xtickangle(45);
try recessionplot; end %#ok<TRYNC>

% Medium (7 vars) -- 3x3, last row centered (var 7 in middle tile of row 3)
figure('Name','Medium (7 vars)','Position',[100 100 1100 700]);
tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for j = 1:6
    nexttile;
    plot(Ydates, Y{2}(:,j), 'r', 'LineWidth', 1.2);
    title(Ynames{2}{j}, 'Interpreter','none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
end
nexttile(8);       % var 7 centered: middle of row 3 (tiles 7,8,9 -> 8)
plot(Ydates, Y{2}(:,7), 'r', 'LineWidth', 1.2);
title(Ynames{2}{7}, 'Interpreter','none');
xtickangle(45);
try recessionplot; end %#ok<TRYNC>

% Large (22 vars) -- 4x6, last row centered (vars 19-22 in cols 2-5)
figure('Name','Large (22 vars)','Position',[50 50 1500 850]);
tiledlayout(4, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
for j = 1:18
    nexttile;
    plot(Ydates, Y{3}(:,j), 'r', 'LineWidth', 1);
    title(Ynames{3}{j}, 'FontSize', 8, 'Interpreter','none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
end
% Row 4: 4 vars centered in tiles 20-23 (tiles 19 and 24 left empty)
for k = 1:4
    nexttile(19 + k);
    plot(Ydates, Y{3}(:,18+k), 'r', 'LineWidth', 1);
    title(Ynames{3}{18+k}, 'FontSize', 8, 'Interpreter','none');
    xtickangle(45);
    try recessionplot; end %#ok<TRYNC>
end
