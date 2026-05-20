clear all;
clc;

% Add path of data and functions
addpath('data');
addpath('functions');

%-------------------------------PRELIMINARIES--------------------------------------
forgetting = 1;    % 1: use constant factor; 2: use variable factor

lambda = 0.99;
kappa = 0.96;      % Decay factor for measurement error variance

eta = 0.99;   % Forgetting factor for DPS (dynamic prior selection) and DMA

% Please choose:
p = 4;             % p is number of lags in the VAR part
nos = 3;           % number of subsets to consider (default is 3, i.e. 3, 7, and 25 variable VARs)
% if nos=1 you might want a single model. Which one is this?
single = 1;        % 1: 3 variable VAR
                   % 2: 7 variable VAR
                   % 3: 25 variable VAR
                   
prior = 1;         % 1: Use Koop-type Minnesota prior
                   % 2: Use Litterman-type Minnesota prior
                         
% Forecasting
first_sample_ends = 1974.75; % The end date of the first sample in the 
                             % recursive exercise (default value: 1969:Q4)
nfore = 8;                   % Select forecast horizon
nsim = 5000;             % Number of times to simulate from the predictive density

% Choose which results to print
% NOTE: CHOOSE ONLY 0/1 (FOR NO/YES) VALUES!
print_fore = 1;           % summary of forecasting results
print_coefficients = 1;   % plot volatilities and lambda_t (but not theta_t which is huge)
print_pred = 1;           % plot predictive likelihoods over time
print_Min = 1;            % print the Minnesota prior over time
%----------------------------------LOAD DATA----------------------------------------
load ydata.dat;
load ynames.mat;
load tcode.dat;
load vars.mat;
%load yearlab.dat;

% Create dates variable
start_date = 1959.00; %1959.Q1
end_date = 2010.25;   %2010.Q2
yearlab = (1959:0.25:2010.25)';
T_thres = find(yearlab == first_sample_ends); % find tau_0 (first sample)

% Transform data to stationarity
% Y: standard transformations (for iterated forecasts, and RHS of direct forecasts)
[Y,yearlab] = transform(ydata,tcode,yearlab);

% Select a subset of the data to be used for the VAR
if nos>3
    error('DMA over too many models, memory concerns...')
end
Y1=cell(nos,1);
Ytemp = standardize1(Y,T_thres);
M = zeros(nos,1);
for ss = 1:nos
    if nos ~= 1
        single = ss;
    end
    select_subset = vars{single,1};
    Y1{ss,1} = Ytemp(:,select_subset);
    M(ss,1) = max(size(select_subset)); % M is the dimensionality of Y
end
t = size(Y1{1,1},1);