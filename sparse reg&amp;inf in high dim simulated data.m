%% Step 1: Simulate high-dimensional data (n=100, p=500, 10 non-zero betas)
rng(12345); % reproducible

n = 100;
p = 500;
s = 10; % number of non-zero coefficients

% Creating a correlated design to mimic imaging-genetics structure:
rho = 0.3;
Sigma = toeplitz(rho.^(0:(p-1))); % Toeplitz covariance (decreasing correlation)
L = chol(Sigma + 1e-6*eye(p),'lower'); % cholesky (small jitter for stability)

% Generate X ~ N(0, Sigma)
Z = randn(n, p);
X = Z * L';

% Creating sparse beta: first s non-zero, others zero 
beta = zeros(p,1);
nonzero_idx = randperm(p, s);
beta(nonzero_idx) = (randn(s,1) .* (2 + rand(s,1))); % moderate effect sizes

% Generating y with noise
sigma_e = 1; % noise std
epsilon = sigma_e * randn(n,1);
y = X * beta + epsilon;

fprintf('Simulated: n=%d, p=%d, nonzeros=%d\n', n, p, s)
fprintf('True nonzero indices (example): '); disp(nonzero_idx(1:min(10,end)))
%% Step 2: Train/test split and standardize
test_frac = 0.3;
n_test = round(n * test_frac);
n_train = n - n_test;

idx = randperm(n);
train_idx = idx(1:n_train);
test_idx  = idx(n_train+1:end);

X_train = X(train_idx, :);
y_train = y(train_idx);
X_test  = X(test_idx, :);
y_test  = y(test_idx);

% Standardizing predictors based on training set
muX = mean(X_train, 1);
sigmaX = std(X_train, 0, 1);
sigmaX(sigmaX==0) = 1; % avoid divide by zero

X_train_std = (X_train - muX) ./ sigmaX;
X_test_std  = (X_test - muX) ./ sigmaX;

% Centering response 
muY = mean(y_train);
y_train_c = y_train - muY;
y_test_c  = y_test - muY;

fprintf('Train/test sizes: %d / %d\n', n_train, n_test)
%% Step 3: LASSO with cross-validation
opts = statset('UseParallel',false,'Display','iter'); 
nLambda = 100;

[BetaLasso, FitInfo] = lasso(X_train_std, y_train_c, ...
    'NumLambda', nLambda, 'CV', 5, 'Options', opts, 'Standardize', false);

% Best lambda from CV
idxLambdaMinMSE = FitInfo.IndexMinMSE;
lambdaMin = FitInfo.Lambda(idxLambdaMinMSE);
beta_lasso_cv = BetaLasso(:, idxLambdaMinMSE);
intercept_lasso = FitInfo.Intercept(idxLambdaMinMSE);

fprintf('LASSO CV chosen lambda = %g (index %d)\n', lambdaMin, idxLambdaMinMSE)
fprintf('Number of nonzeros in LASSO (CV) = %d\n', sum(beta_lasso_cv ~= 0))
%% Step 4: Plotting coefficient path and CV MSE
figure(1); clf;
lassoPlot(BetaLasso, FitInfo, 'PlotType','Lambda','XScale','log');
title('LASSO coefficient paths');

figure(2); clf;
semilogx(FitInfo.Lambda, FitInfo.MSE, '-o');
hold on;
xline(lambdaMin, '--r', 'LineWidth', 1.5);
xlabel('\lambda'); ylabel('CV MSE');
set(gca,'XScale','log');
title('LASSO CV MSE vs Lambda');
legend('CV MSE','Chosen \lambda');
grid on;
%% Step 5: Fitting Ridge, Elastic Net, and fitrlinear-LASSO (with CV)
% Ridge (L2)
rng(0);
mdlRidge = fitrlinear(X_train_std, y_train_c, ...
    'Learner', 'leastsquares', 'Regularization','ridge', ...
    'KFold',5, 'Lambda', logspace(-6,2,30));

% Elastic Net: trying several Alpha values and pick best via nested CV (simple grid)
alpha_grid = [0.2, 0.5, 0.8]; % 0=ridge,1=lasso
bestEN = struct('Alpha',[],'Lambda',[],'Model',[],'Loss',Inf);

for a = alpha_grid
    mdl = fitrlinear(X_train_std, y_train_c, ...
        'Learner','leastsquares','Regularization','elasticnet','Alpha',a, ...
        'KFold',5,'Lambda',logspace(-6,1,20));
    loss = kfoldLoss(mdl);
    if loss < bestEN.Loss
        bestEN.Alpha = a;
        bestEN.Lambda = mdl.ModelParameters.Lambda;
        bestEN.Model = mdl;
        bestEN.Loss = loss;
    end
end
fprintf('Best Elastic Net alpha = %.2f, CV loss = %.4f\n', bestEN.Alpha, bestEN.Loss)

% fitrlinear-LASSO (for comparison)
mdlLassoFL = fitrlinear(X_train_std, y_train_c, ...
    'Learner','leastsquares','Regularization','lasso','KFold',5,'Lambda',logspace(-6,0,30));

% Extracting cross-validated models or final trained models on all data:
% For simplicity, i choose to train final fitrlinear models on full training set using best lambda found.
% Ridge final: choosing lambda with lowest CV loss from mdlRidge
[~, idxR] = min(kfoldLoss(mdlRidge, 'Mode', 'individual'));
% Actually, easier: refit a ridge using built-in crossval tuning via fitrlinear with 'OptimizeHyperparameters' if available.
% Note: Since LASSO regularization path could not be computed due to license restrictions,
% an approximate value of lambda = 1e-2 is used here for demonstration purposes
lambda_ridge_final = 1e-2;
mdlRidgeFinal = fitrlinear(X_train_std, y_train_c, 'Learner','leastsquares','Regularization','ridge','Lambda',lambda_ridge_final);

% Elastic Net final using bestEN.Alpha and median Lambda from the internal model:
lambda_en_final = median(bestEN.Model.ModelParameters.Lambda);
mdlENFinal = fitrlinear(X_train_std, y_train_c, 'Learner','leastsquares','Regularization','elasticnet','Alpha',bestEN.Alpha,'Lambda',lambda_en_final);

% LASSO final via fitrlinear: median lambda from mdlLassoFL crossval
lambda_lasso_final = median(mdlLassoFL.ModelParameters.Lambda);
mdlLassoFinal = fitrlinear(X_train_std, y_train_c,'Learner','leastsquares','Regularization','lasso','Lambda',lambda_lasso_final);

fprintf('Fitted final ridge, elastic net, lasso (fitrlinear) models.\n')
%% Step 6: Evaluating on test set
% Predictions:
yhat_lasso = X_test_std * beta_lasso_cv + intercept_lasso; % lasso from step 3 (y was centered)
% adding back training mean
yhat_lasso = yhat_lasso + muY;

% fitrlinear predictions (they were trained on centered y_train_c). so adding muY back:
yhat_ridge = predict(mdlRidgeFinal, X_test_std) + muY;
yhat_en    = predict(mdlENFinal, X_test_std) + muY;
yhat_lf    = predict(mdlLassoFinal, X_test_std) + muY;

% True y_test (not centered)
y_true = y_test;

% Functions
mse = @(y,yhat) mean((y-yhat).^2);
rsq = @(y,yhat) 1 - sum((y-yhat).^2)/sum((y-mean(y)).^2);

MSEs = [mse(y_true, yhat_lasso), mse(y_true, yhat_ridge), mse(y_true, yhat_en), mse(y_true, yhat_lf)];
R2s  = [rsq(y_true, yhat_lasso), rsq(y_true, yhat_ridge), rsq(y_true, yhat_en), rsq(y_true, yhat_lf)];

modelNames = {'LASSO (lasso)','Ridge (fitrlinear)','ElasticNet (fitrlinear)','LASSO (fitrlinear)'};

T = table(modelNames', MSEs', R2s', 'VariableNames', {'Model','MSE','R2'});
disp(T);

% Plot
figure(3); clf;
bar(categorical(modelNames), MSEs);
ylabel('Test MSE');
title('Model test MSE comparison');

figure(4); clf;
bar(categorical(modelNames), R2s);
ylabel('Test R^2');
title('Model test R^2 comparison');
%% Step 7: Debiased LASSO (approximate) and p-values
% Using training standardized X_train_std and lasso beta from step 3 (beta_lasso_cv)
Xn = X_train_std;
yn = y_train_c;
n_tr = size(Xn,1);

% Computing sample covariance and regularize
Sigma_hat = (Xn' * Xn) / n_tr;
tau = 1e-2; % regularization parameter 
Sigma_reg = Sigma_hat + tau * eye(p);

% Invert 
Sigma_inv_approx = pinv(Sigma_reg); % pseudoinverse for numerical stability

% Computing residuals using lasso fit
resid = yn - Xn * beta_lasso_cv - intercept_lasso; % intercept_lasso is from step 3 and applied to centered y
% Debiased estimate
beta_debiased = beta_lasso_cv + (1/n_tr) * (Sigma_inv_approx * (Xn' * resid));

% Estimating noise variance (using degrees of freedom approximation)
sigma2_hat = sum(resid.^2) / (n_tr - 1); % rough

% Standard errors approximate: se_j = sqrt( sigma2_hat * diag(Sigma_inv_approx) / n )
se_debias = sqrt( sigma2_hat * diag(Sigma_inv_approx) / n_tr );

% z-scores and p-values (two-sided)
z_scores = beta_debiased ./ se_debias;
pvals = 2 * (1 - normcdf(abs(z_scores))); % normcdf from Statistics toolbox

% Show top hits by p-value
[~, order] = sort(pvals);
topk = 20;
fprintf('Top %d variables by approximate p-value:\n', topk)
disp(table(order(1:topk), beta_debiased(order(1:topk)), se_debias(order(1:topk)), pvals(order(1:topk)), ...
    'VariableNames', {'Index','BetaDebiased','SE','pValue'}));

