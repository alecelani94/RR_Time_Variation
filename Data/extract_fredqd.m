clear
close all
clc

% =========================================================================
% Extract FRED-QD datasets (Small_3 / Small_4 / Small_5 / Medium / Large)
%
% Reads the most recent *-QD.csv in this folder, applies GLP transformations
% (400 * Delta log for growth series, raw for rates / utilization), and
% saves one .mat per dataset. Set do_plot = true to also visualize them.
% =========================================================================

do_plot = false;   % set true to draw one figure per dataset

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

small_3_vars = {
    'GDPC1',        'RGDP',             '4log'
    'GDPCTPI',      'PGDP',             '4log'
    'FEDFUNDS',     'FedFunds',         'raw'
};

small_4_vars = [small_3_vars; {
    'UNRATE',       'UnempRate',        'raw'
}];

small_5_vars = [small_4_vars; {
    'INDPRO',       'IPgrowth',         '4log'
}];

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

datasets = {small_3_vars, small_4_vars, small_5_vars, medium_vars, large_vars};
dsnames  = {'Small_3', 'Small_4', 'Small_5', 'Medium', 'Large'};
nDS    = numel(datasets);
Y      = cell(1, nDS);
Ynames = cell(1, nDS);

for d = 1:nDS
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

out_files = {'fredqd_small_3.mat', 'fredqd_small_4.mat', 'fredqd_small_5.mat', ...
             'fredqd_medium.mat', 'fredqd_large.mat'};
for d = 1:nDS
    out_path = fullfile(here, out_files{d});
    data   = Y{d};
    names  = Ynames{d};
    dates_ = Ydates;
    save(out_path, 'data', 'names', 'dates_');
    fprintf('Saved %-22s  %d x %2d\n', out_files{d}, size(Y{d},1), size(Y{d},2));
end

%% ----- Plots ------------------------------------------------------------
% Delegate to Functions/plot_dataset.m so layouts stay in one place.

if do_plot
    addpath(fullfile(fileparts(here), 'Functions'));
    for d = 1:nDS
        plot_dataset(Y{d}, Ydates, Ynames{d});
    end
end
