function [Y, Names, Ydates, T, N] = load_slice(dataset, date_start, date_end)
% LOAD_SLICE  Load a FRED-QD dataset (small / medium / large) and slice
% it by the requested date window. Echoes a one-line summary.
%
% Inputs
%   dataset    : 'small' | 'medium' | 'large'  (chooses Data/fredqd_<dataset>.mat)
%   date_start : datetime, inclusive (first available 1959-Q2)
%   date_end   : datetime, inclusive (last available depends on the .mat)
%
% Outputs
%   Y      : T x N data
%   Names  : 1 x N cell of variable labels
%   Ydates : T x 1 datetime (quarterly stamps)
%   T, N   : sample length and number of variables

datafile = fullfile('Data', ['fredqd_' lower(dataset) '.mat']);
if ~isfile(datafile)
    error('Dataset file not found: %s (run Data/extract_fredqd.m first).', datafile);
end
S = load(datafile, 'data', 'names', 'dates_');

% accept either native datetime (MATLAB extraction) or string (Python extraction)
if isdatetime(S.dates_)
    Ydates = S.dates_;
else
    Ydates = datetime(string(S.dates_));
end
Ydates = Ydates(:);

Y     = S.data;
Names = S.names;

mask = (Ydates >= date_start) & (Ydates <= date_end);
if ~any(mask)
    error('No observations in [%s, %s].', string(date_start), string(date_end));
end
Y      = Y(mask, :);
Ydates = Ydates(mask);
[T, N] = size(Y);

fprintf('Dataset:   %s (%d variables)\n', dataset, N);
fprintf('Sample:    %s to %s (%d quarters)\n', ...
    string(Ydates(1)), string(Ydates(end)), T);
fprintf('Variables: %s\n', strjoin(Names, ', '));
end
