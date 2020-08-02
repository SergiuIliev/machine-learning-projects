%%% Sergiu Iliev | Bayesian ML Course | Final | Problem 2 - continued
clc, clear

%% Load dataset to continue from Python
T = readtable('DF.csv');           % Create a temporary table to import the data

y = T.Energy;                      % Define the Energy vector to be predicted (label) 

T = removevars(T,{'Date','Energy'}); % Keep only the predictor variables in the table
X = table2array(T);                  % Create the predictor array
predictornames = T.Properties.VariableNames(); % Create a vector of the predictor names

figure;
boxplot(X);
title('Variable Box Plots');

% T = readtable('DF.csv');           % Regenerate the temporary table
% DF = transpose(load('DF.mat'));  % load dataset
% X = transpose(load('X.mat'));    % load training dataset
% y = transpose(load('y.mat'));    % load testing dataset
% fnames = fieldnames(Y);
% T = table;
% for i = 1:length(fnames)
%     x_T = table(num2cell(getfield(Y, fnames{i})));
%     x_T.Properties.VariableNames = {fnames{i}};
%     T = [T, x_T];
% end   
%% (d)-continued: Constructing a Bayesian Linear Regression Model with a LASSO prior
rng('default')                      % For reproducibility

lambda = 1;                         % Specify a regularization value of 1

B = lasso(X,y,'Lambda',lambda, 'PredictorNames',predictornames) % Construct the posterior linear regression model using the data.
% [LassoBetaEstimates,FitInfo] = lasso(X,y,'PredictorNames',predictornames);

plot(B)                             % Plot the posterior probability distribution
%% (d) 2nd Variant
p = size(X,2); % Number of predictors

PriorMdl = bayeslm(p,'ModelType','lasso','VarNames',predictornames);
table(PriorMdl.Lambda,'RowNames',PriorMdl.VarNames)



%%
% Compute the FMSE of each model returned by lasso.
yFLasso = FitInfo.Intercept + XF*LassoBetaEstimates;
fmseLasso = sqrt(mean((yF - yFLasso).^2,1));

% Plot the magnitude of the regression coefficients with respect to the shrinkage value.
hax = lassoPlot(LassoBetaEstimates,FitInfo);
L1Vals = hax.Children.XData;
yyaxis right
h = plot(L1Vals,fmseLasso,'LineWidth',2,'LineStyle','--');
legend(h,'FMSE','Location','SW');
ylabel('FMSE');
title('Frequentist Lasso')
%%
fmsebestlasso = min(fmseLasso(FitInfo.DF == 6));
idx = fmseLasso == fmsebestlasso;
bestLasso = [FitInfo.Intercept(idx); LassoBetaEstimates(:,idx)];
table(bestLasso,'RowNames',["Intercept" predictornames])

%%
The frequentist lasso analysis suggests that the variables CPIAUCSL, GCE, GDP, GPDI, PCEC, and FEDFUNDS are either insignificant or redundant.



%% (e) Constructing and Plotting the Posterior
%% Variable Selection

V1 = [1 100 1000];

PriorMdl = lasso(X,y,'Lambda',lambda, 'PredictorNames',predictornames)

V = [V1(j)*ones(p + 1,1) V2(k)*ones(p + 1,1)];
PriorMdl{j,k} = bayeslm(p,'ModelType','mixconjugateblm',...
            'VarNames',predictornames,'V',V);
