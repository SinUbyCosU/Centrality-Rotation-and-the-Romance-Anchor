import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def compute_cohens_d(group1, group2):
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("undefined_empty_group")
    
    var1 = np.var(group1, ddof=1) if len(group1) > 1 else 0
    var2 = np.var(group2, ddof=1) if len(group2) > 1 else 0
    
    if var1 == 0 and var2 == 0:
        raise ValueError("undefined_zero_variance")
        
    pooled_std = np.sqrt(((len(group1)-1)*var1 + (len(group2)-1)*var2) / (len(group1) + len(group2) - 2))
    if pooled_std == 0:
        raise ValueError("undefined_zero_variance")
        
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def compute_loo_r2(X, y):
    if np.var(y) == 0:
        raise ValueError("undefined_zero_variance")
    
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds = []
    
    # Check if X is 1D, reshape if necessary
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
        
    for train_idx, test_idx in loo.split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        preds.append(model.predict(X[test_idx])[0])
        
    return r2_score(y, preds)

def compute_bootstrap_r2(X, y, n_boot=1000, random_state=None):
    if np.var(y) == 0:
        raise ValueError("undefined_zero_variance")
        
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
        
    if random_state is not None:
        np.random.seed(random_state)
        
    boot_r2 = []
    for _ in range(n_boot):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_b = X[idx]
        y_b = y[idx]
        
        if len(np.unique(y_b)) > 1:
            r_b = LinearRegression().fit(X_b, y_b)
            boot_r2.append(r2_score(y_b, r_b.predict(X_b)))
            
    if not boot_r2:
        raise ValueError("undefined_zero_variance")
        
    boot_mean = np.mean(boot_r2)
    boot_ci = (np.percentile(boot_r2, 2.5), np.percentile(boot_r2, 97.5))
    return boot_mean, boot_ci
