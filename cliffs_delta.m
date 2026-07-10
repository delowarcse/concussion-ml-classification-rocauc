function [delta, stats] = cliffs_delta(x, y, nboot, alpha, seed)
%CLIFFS_DELTA Cliff's delta for two independent samples.
%
%   delta = P(X > Y) - P(X < Y), with ties contributing 0.
%
% Inputs:
%   x, y   : vectors containing observations from two independent groups
%   nboot  : number of bootstrap resamples for CI; 0 = no CI
%   alpha  : alpha level, e.g., 0.05 for 95% CI
%   seed   : optional random seed. Leave empty if rng was already set.
%
% Outputs:
%   delta  : Cliff's delta in [-1, 1]
%   stats  : struct with n1, n2, medX, iqrX, medY, iqrY, A, and ci

    if nargin < 3 || isempty(nboot), nboot = 0; end
    if nargin < 4 || isempty(alpha), alpha = 0.05; end
    if nargin < 5, seed = []; end

    % Ensure column vectors and remove missing values
    x = x(:);
    y = y(:);

    x = x(~isnan(x));
    y = y(~isnan(y));

    n1 = numel(x);
    n2 = numel(y);

    if n1 == 0 || n2 == 0
        error('cliffs_delta:EmptyInput', ...
            'x and y must contain at least one non-NaN value each.');
    end

    % Descriptive statistics
    medX = median(x);
    iqrX = iqr(x);
    medY = median(y);
    iqrY = iqr(y);

    % Pairwise comparisons
    % D is an n1-by-n2 matrix containing all pairwise differences.
    D = x - y.';

    greater = sum(D(:) > 0);
    less    = sum(D(:) < 0);
    ties    = n1*n2 - greater - less;

    % Cliff's delta
    delta = (greater - less) / (n1*n2);

    % Vargha-Delaney A statistic:
    % probability that a randomly selected X observation is larger than a
    % randomly selected Y observation, with ties counted as half.
    A = (greater + 0.5*ties) / (n1*n2);

    % Store results
    stats = struct('n1', n1, 'n2', n2, 'medX', medX, 'iqrX', iqrX, 'medY', medY, 'iqrY', iqrY, 'A', A, 'ci', [NaN, NaN]);

    % Bootstrap confidence interval
    if nboot > 0

        % Only reset rng if a seed is explicitly provided.
        % Otherwise, use the global rng state set before the loop.
        if ~isempty(seed)
            rng(seed);
        end

        boot = zeros(nboot,1);

        for b = 1:nboot
            xb = x(randi(n1, n1, 1));
            yb = y(randi(n2, n2, 1));

            Db = xb - yb.';
            gb = sum(Db(:) > 0);
            lb = sum(Db(:) < 0);

            boot(b) = (gb - lb) / (numel(xb) * numel(yb));
        end

        q = 100 * [alpha/2, 1 - alpha/2];
        stats.ci = prctile(boot, q);

    end

end