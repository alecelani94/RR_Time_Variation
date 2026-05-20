% Process raw data from the news shocks set (taken from Luca's code)
X = xlsread( 'newsshocks.xls', 'QuarterlyData', 'C44:W261' );
[T, N] = size( X );
Time = X( 2:T, 1 );
LogGDPDeflator = log( X( :, 2 ) ); Inflation=diff( LogGDPDeflator );
LogRealGDP = log( X( :, 3 ) );
LogPopulation = log( X( :, 4 ) );
LogRealGDPPerCapita = LogRealGDP - LogPopulation;
LogHoursWorkedPerCapita = log( X( :, 5 ) ) - LogPopulation;
FEDFUNDS = ( X( :, 6 ) / 100 + 1 ) .^ ( 1 / 4 ) - 1;
TB3MS = ( X( :, 7 ) / 100 + 1 ) .^ ( 1 / 4 ) - 1;
GS1 = ( X( :, 8) / 100 + 1 ) .^ ( 1 / 4 ) - 1;
GS3 = ( X( :, 9) / 100 + 1 ) .^ ( 1 / 4 ) - 1;
GS5 = ( X( :, 10) / 100 + 1 ) .^ ( 1 / 4 ) - 1;
GS10 = ( X( :, 11) / 100 + 1 ) .^ ( 1 / 4 ) - 1;
LogTFP = cumsum( X( :, 12 ) / 400 );
LogRPI = X( :, 13 );
LogLaborProductivity = log( X( :, 14 ) );
LogRealSEP500 = log( X( :, 15 ) ) - LogGDPDeflator;
LogRealDividends = log( X( :, 16 ) ) - LogGDPDeflator;
LogRealConsumptionPerCapita = X( :, 17 ) - LogPopulation;
LogRealInvestmentPerCapita = X( :, 18 ) - LogPopulation;
UnemploymentRate = X( :, 19 );
VacancyRate = X( :, 20 );
LogRealDefenseExpPerCapita = log( X( :, 21 ) ) - LogPopulation;

X1 = [LogTFP( 2:T ) FEDFUNDS( 2:T ) Inflation LogHoursWorkedPerCapita( 2:T ) / 4 ...
      LogRealGDPPerCapita( 2:T ) LogRealConsumptionPerCapita( 2:T ) ...
      LogRealInvestmentPerCapita( 2:T ) GS5( 2:T ) LogRPI( 2:T ) LogRealSEP500( 2:T ) / 400 ...
      UnemploymentRate( 2:T ) / 4 VacancyRate(2:T) / 400 ...
      TB3MS( 2:T ) GS10( 2:T ) LogRealDividends( 2:T ) / 400 ...
      GS1( 2:T ) GS3( 2:T )  LogLaborProductivity( 2:T ) LogRealDefenseExpPerCapita( 2:T )] * 400;

% Process raw data from the FRED-QD set
X2 = [];

csvwrite( 'newsshocks01.csv', [X1 X2] );

%%% OLD CONSTRUCTION %%%
% X1 = [LogTFP( 2:T ) FEDFUNDS( 2:T ) Inflation LogHoursWorkedPerCapita( 2:T ) / 4 ...
%       LogRealGDPPerCapita( 2:T ) LogRealConsumptionPerCapita( 2:T ) ...
%       LogRealInvestmentPerCapita( 2:T ) LogRPI( 2:T ) UnemploymentRate( 2:T ) / 4 ...
%       TB3MS( 2:T ) GS1( 2:T ) GS3( 2:T ) GS5( 2:T ) GS10( 2:T ) LogRealSEP500( 2:T ) / 400 ...
%       VacancyRate(2:T) / 400 LogLaborProductivity( 2:T ) LogRealDividends( 2:T ) / 400] * 400;