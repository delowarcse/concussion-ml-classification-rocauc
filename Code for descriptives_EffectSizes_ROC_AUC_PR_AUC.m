%% -------------------------------------------------------------------------
%  Cohen's d effect size calculation
% -------------------------------------------------------------------------
%% Read CSV file

filename = 'S2P_Data.csv';

T = readtable(filename, ...
    'VariableNamingRule', 'preserve');

%% Extract numeric data from table

% Rows 1:219 = No history of concussion
% Rows 220:320 = History of concussion
% Columns 14:92 = robotic parameters

x = T{1:219, 14:92}; %No history of concussion. Columns 14-92 contain the robotic parameters
y = T{220:320, 14:92}; %With history of concussion. Columns 14-92 contain the robotic parameters

assert(isnumeric(x), 'x must be numeric. Check whether columns 14:92 contain text or categorical values.');
assert(isnumeric(y), 'y must be numeric. Check whether columns 14:92 contain text or categorical values.');

%%
% This section computes Cohen's d for independent-group comparisons.
%
% For each variable, Cohen's d is calculated as:
%
%       d = (mean_x - mean_y) / pooled_sd
%
% where x and y are the two independent groups and pooled_sd is calculated as:
%
%       pooled_sd = sqrt( ((n_x - 1)*sd_x^2 + (n_y - 1)*sd_y^2) ...
%                         / (n_x + n_y - 2) )
%
% Means and standard deviations are calculated column-wise. Missing values,
% if present, are ignored. Sample size is calculated from the first column of
% each group.
%
% The sign of Cohen's d is directional:
%   d > 0 indicates higher values in group x than group y
%   d < 0 indicates lower values in group x than group y
%
% This calculation assumes independent samples and uses the pooled standard
% deviation. It does not apply a small-sample correction. Thus, the
% reported values correspond to Cohen's d rather than Hedges' g.
% -------------------------------------------------------------------------

% Compute column-wise means and SDs, ignoring NaNs
mean_x = mean(x, 1, 'omitnan');
sd_x   = std(x, 0, 1, 'omitnan');

mean_y = mean(y, 1, 'omitnan');
sd_y   = std(y, 0, 1, 'omitnan');

% Sample sizes
% This assumes the same number of valid observations for all columns.
n_x = sum(~isnan(x(:,1)));
n_y = sum(~isnan(y(:,1)));

% Compute pooled SD and Cohen's d for independent samples
pooled_sd = sqrt(((n_x - 1) * sd_x.^2 + (n_y - 1) * sd_y.^2) / ...
                 (n_x + n_y - 2));

cohens_d = (mean_x - mean_y) ./ pooled_sd;

% Disable scientific notation
format longG

% Format mean ± SD and Cohen's d as fixed-point strings
summary_x = strings(1, size(x,2));
summary_y = strings(1, size(y,2));
summary_d = strings(1, size(x,2));

for i = 1:size(x,2)
    summary_x(i) = sprintf('%.4f ± %.4f', mean_x(i), sd_x(i));
    summary_y(i) = sprintf('%.4f ± %.4f', mean_y(i), sd_y(i));
    summary_d(i) = sprintf('%.2f', cohens_d(i));
end

% Combine into one table:
%   Column 1 = No history of concussion (mean ± SD)
%   Column 2 = With history of concussion (mean ± SD)
%   Column 3 = Cohen's d
summary_table = [summary_x' summary_y' summary_d'];

clc


%% -------------------------------------------------------------------------
%  Directional univariate ROC AUC calculation for robotic parameters
% -------------------------------------------------------------------------
% This section computes the ROC AUC for each feature individually.
%
% Each column of X is used directly as a continuous classification score.
% The true class labels are provided in y, with 1 indicating the
% positive class.
%
% AUCs(j) reflects the ability of feature j alone to discriminate between
% participants in the positive and negative classes, using the original
% direction of the feature values.
%
% Directional interpretation:
%   - AUC > 0.5 indicates that higher values of the feature tend to be
%     associated with the positive class.
%   - AUC < 0.5 indicates that higher values of the feature tend to be
%     associated with the negative class.
% 
%
% MATLAB's built-in perfcurve function is used to compute the ROC AUC.
% -------------------------------------------------------------------------
%% Load robotic data
filename = 'S2P_Data.csv';

T = readtable(filename, ...
    'VariableNamingRule', 'preserve');

X = T{:,14:92}; %columns 14-92 contain the robotic parameters
y = T{:,1}; %column 1 encodes history of concussion. 0 = No history (n = 219), 1 = With history (n = 101);

assert(isnumeric(X), 'x must be numeric. Check whether columns 14:92 contain text or categorical values.');
assert(isnumeric(y), 'y must be numeric. Check whether columns 14:92 contain text or categorical values.');

%% Run ROC on robotic data 

nVars = size(X, 2);
AUCs  = nan(nVars, 1);

for j = 1:nVars
    [~, ~, ~, AUCs(j)] = perfcurve(y, X(:, j), 1);
end

Parameter = categorical(T.Properties.VariableNames(14:92))';
AUCs = round(AUCs, 3);

AUC_table = table(Parameter, AUCs, ...
    'VariableNames', {'Parameter', 'ROC_AUC'});

disp(AUC_table)

clear AUCs j nVars Parameter T

%% -------------------------------------------------------------------------
%  Directional univariate ROC AUC calculation for clinical parameters
% -------------------------------------------------------------------------
% This section computes the ROC AUC for each feature individually.
%
% Each column of X is used directly as a continuous classification score.
% The true class labels are provided in y, with 1 indicating the
% positive class.
%
% AUCs(j) reflects the ability of feature j alone to discriminate between
% participants in the positive and negative classes, using the original
% direction of the feature values.
%
% Directional interpretation:
%   - AUC > 0.5 indicates that higher values of the feature tend to be
%     associated with the positive class.
%   - AUC < 0.5 indicates that higher values of the feature tend to be
%     associated with the negative class.
%   - AUC values were not transformed using max(AUC, 1-AUC), because this
%     would remove directionality and would not be directly comparable to
%     the directional PR AUC analysis.
%
% MATLAB's built-in perfcurve function is used to compute the ROC AUC.
% -------------------------------------------------------------------------
%% Load clinincal data
filename = 'S2P_Data.csv';

T = readtable(filename, ...
    'VariableNamingRule', 'preserve');

X = T{:,2:13}; %columns 2-13 contain the clinical variables
y = T{:,1}; %column 1 encodes history of concussion. 0 = No history (n = 219), 1 = With history (n = 101);

assert(isnumeric(X), 'x must be numeric. Check whether columns 2:13 contain text or categorical values.');
assert(isnumeric(y), 'y must be numeric. Check whether columns 2:13 contain text or categorical values.');

%% Run ROC on clinical data 

nVars = size(X, 2);
AUCs  = nan(nVars, 1);

for j = 1:nVars
    [~, ~, ~, AUCs(j)] = perfcurve(y, X(:, j), 1);
end

Parameter = categorical(T.Properties.VariableNames(2:13))';
AUCs = round(AUCs, 3);

AUC_table = table(Parameter, AUCs, ...
    'VariableNames', {'Parameter', 'ROC_AUC'});

disp(AUC_table)

clear AUCs j nVars Parameter T

%% -------------------------------------------------------------------------
%  Precision-recall AUC calculation for robotic parameters
% --------------------------------------------------------------------------
% This section computes the precision-recall AUC for each feature
% individually.
%
% Each column of X is used directly as a continuous classification score.
% The true class labels are provided in labels, with 1 indicating the
% positive class.
%
% AUPRCs(j) reflects the precision-recall AUC for feature j alone, using the
% original direction of the feature values.
%
% Directional interpretation:
%   - Higher feature values are treated as stronger evidence for the positive
%     class.
%   - If lower feature values are associated with the positive class, the
%     directional PR AUC may be low.
%
% The no-skill PR AUC baseline is equal to the prevalence of
% the positive class. With 101 of 320 observations belonging to the
% positive class, the no-skill baseline is:
%
%       101 / 320 = 0.316
%
% Therefore, PR AUC values near the positive-class prevalence indicate
% near-baseline performance, whereas values below prevalence indicate
% performance below the no-skill baseline in the specified score direction.
%
% MATLAB's built-in perfcurve function is used to compute the PR AUC by
% specifying recall on the x-axis and precision on the y-axis.
% -------------------------------------------------------------------------

%% Load robotic data
filename = 'S2P_Data.csv';

T = readtable(filename, ...
    'VariableNamingRule', 'preserve');

X = T{:,14:92}; %columns 14-92 contain the robotic parameters
y = T{:,1}; %column 1 encodes history of concussion. 0 = No history (n = 219), 1 = With history (n = 101);

assert(isnumeric(X), 'x must be numeric. Check whether columns 14:92 contain text or categorical values.');
assert(isnumeric(y), 'y must be numeric. Check whether columns 14:92 contain text or categorical values.');

%% run PR AUC analysis on robotic data 
nVars  = size(X, 2);
AUPRCs = nan(nVars, 1);

for j = 1:nVars
    [rec, prec, ~, AUPRCs(j)] = perfcurve(y, X(:, j), 1, ...
        'xCrit', 'reca', ...
        'yCrit', 'prec');
end

Parameter = categorical(T.Properties.VariableNames(14:92))';
AUPRCs = round(AUPRCs, 3);

AUPRC_table = table(Parameter, AUPRCs, ...
    'VariableNames', {'Parameter', 'PR_AUC'});

disp(AUPRC_table)

clear AUPRCs j nVars Parameter T

%% -------------------------------------------------------------------------
%  Precision-recall AUC calculation for clinical parameters
% --------------------------------------------------------------------------
% This section computes the precision-recall AUC for each feature
% individually.
%
% Each column of X is used directly as a continuous classification score.
% The true class labels are provided in labels, with 1 indicating the
% positive class.
%
% AUPRCs(j) reflects the precision-recall AUC for feature j alone, using the
% original direction of the feature values.
%
% Directional interpretation:
%   - Higher feature values are treated as stronger evidence for the positive
%     class.
%   - If lower feature values are associated with the positive class, the
%     directional PR AUC may be low.
%
% The no-skill PR AUC baseline is equal to the prevalence of
% the positive class. With 101 of 320 observations belonging to the
% positive class, the no-skill baseline is:
%
%       101 / 320 = 0.316
%
% Therefore, PR AUC values near the positive-class prevalence indicate
% near-baseline performance, whereas values below prevalence indicate
% performance below the no-skill baseline in the specified score direction.
%
% MATLAB's built-in perfcurve function is used to compute the PR AUC by
% specifying recall on the x-axis and precision on the y-axis.
% -------------------------------------------------------------------------

%% Load clinical data
filename = 'S2P_Data.csv';

T = readtable(filename, ...
    'VariableNamingRule', 'preserve');

X = T{:,2:13}; %columns 2-13 contain the clinical parameters
y = T{:,1}; %column 1 encodes history of concussion. 0 = No history (n = 219), 1 = With history (n = 101);

assert(isnumeric(X), 'x must be numeric. Check whether columns 2:13 contain text or categorical values.');
assert(isnumeric(y), 'y must be numeric. Check whether columns 2:13 contain text or categorical values.');

%% run PR AUC analysis on clinical parameters 
nVars  = size(X, 2);
AUPRCs = nan(nVars, 1);

for j = 1:nVars
    [rec, prec, ~, AUPRCs(j)] = perfcurve(y, X(:, j), 1, ...
        'xCrit', 'reca', ...
        'yCrit', 'prec');
end

Parameter = categorical(T.Properties.VariableNames(2:13))';
AUPRCs = round(AUPRCs, 3);

AUPRC_table_Clinical = table(Parameter, AUPRCs, ...
    'VariableNames', {'Parameter', 'PR_AUC'});

disp(AUPRC_table_Clinical)

clear AUPRCs j nVars Parameter T

%% ============================================================
%  Cliff's delta implementation
% ============================================================
% This script computes Cliff's delta for each nominal and ordinal variable/feature.
%
% Cliff's delta is a non-parametric effect size for comparing two
% independent groups. It is calculated as:
%
%       delta = P(X > Y) - P(X < Y)
%
% where X and Y are observations from two independent groups. Values range
% from -1 to +1:
%
%       delta > 0 indicates larger values in group X than group Y
%       delta < 0 indicates smaller values in group X than group Y
%       delta = 0 indicates no directional dominance between groups
%
% Ties contribute zero to Cliff's delta.
%
% This script assumes:
%   - X is an n1-by-p matrix
%   - Y is an n2-by-p matrix
%   - Rows are observations/participants
%   - Columns are ordinal variables/features
%   - X and Y contain the same number of columns
%
% For each variable, the script returns:
%   - sample size for each group
%   - median and interquartile range for each group
%   - Cliff's delta
%   - bootstrap confidence interval for Cliff's delta
%
% The local function cliffs_delta must be included at the bottom of this
% script or saved as a separate function on the MATLAB path.

%% ======================================= 
% Read CSV file

filename = 'S2P_Data.csv';

T = readtable(filename, ...
    'VariableNamingRule', 'preserve');

%% Extract numeric data from table

% Rows 1:219 = No history of concussion
% Rows 220:320 = History of concussion
% Columns 2:13 = clinical variables/features of interest

X = T{1:219, 2:13};
Y = T{220:320, 2:13};

assert(isnumeric(X), 'x must be numeric. Check whether columns 2:13 contain text or categorical values.');
assert(isnumeric(Y), 'y must be numeric. Check whether columns 2:13 contain text or categorical values.');
%% ============================================================

% ---- sanity checks ----
assert(size(X,2) == size(Y,2), ...
    'X and Y must have the same number of columns (variables).');

p = size(X,2);

% ---- settings ----
nboot = 5000;   % number of bootstrap resamples; set to 0 to skip CIs
alpha = 0.05;   % alpha level for 95% confidence intervals
seed  = 1;      % random seed for reproducibility

% Set random seed once before all bootstrap calculations
rng(seed);

% ---- results table ----
res = table('Size',[p 10], ...
    'VariableTypes', {'double','double','double','double','double', ...
                      'double','double','double','double','double'}, ...
    'VariableNames', {'var','n1','n2','medX','iqrX','medY','iqrY', ...
                      'delta','ci_low','ci_high'});

% ---- compute Cliff's delta for each variable ----
for j = 1:p

    % Extract variable j from each group
    xj = X(:,j);
    yj = Y(:,j);

    % Compute Cliff's delta and summary statistics.
    % The seed argument is left empty because rng(seed) was set once above.
    [d, st] = cliffs_delta(xj, yj, nboot, alpha, []);

    % Store results
    res.var(j) = j; 
    res.n1(j) = st.n1; 
    res.n2(j) = st.n2; 
    res.medX(j) = st.medX; 
    res.iqrX(j) = st.iqrX; 
    res.medY(j) = st.medY; 
    res.iqrY(j) = st.iqrY; 
    res.delta(j) = d; 
    res.ci_low(j) = st.ci(1); 
    res.ci_high(j) = st.ci(2);

end

% Round displayed numeric results to 3 decimal places
res.medX    = round(res.medX, 3);
res.iqrX    = round(res.iqrX, 3);
res.medY    = round(res.medY, 3);
res.iqrY    = round(res.iqrY, 3);
res.delta   = round(res.delta, 3);
res.ci_low  = round(res.ci_low, 3); 
res.ci_high = round(res.ci_high, 3); 

% Display results table
disp(res)