"""
Task 1: The Inference Engine - Custom OLS Regression
面向对象实现，支持多实例独立运行
"""

import numpy as np
from scipy import stats
from typing import Dict


class CustomOLS:
    """
    Custom OLS Regression Engine
    手写实现最小二乘回归，包含完整的统计推断功能
    """

    def __init__(self, fit_intercept: bool = True):
        """
        初始化模型

        Parameters:
        -----------
        fit_intercept : bool
            是否添加截距项（默认True）
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.residuals_ = None
        self.fitted_values_ = None
        self._X_design = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """添加截距列（全1列）"""
        if not self.fit_intercept:
            return X
        n = X.shape[0]
        return np.column_stack([np.ones(n), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合模型，计算 β̂, σ̂², 协方差矩阵

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵，shape (n_samples, n_features)
        y : np.ndarray
            目标变量，shape (n_samples,)

        Returns:
        --------
        self : 返回自身，支持链式调用
        """
        # 添加截距项
        X_design = self._add_intercept(X)
        self._X_design = X_design

        n, p = X_design.shape

        # 1. 计算 β̂ = (XᵀX)⁻¹ Xᵀ y
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)
        XtY = X_design.T @ y
        self.coef_ = XtX_inv @ XtY

        # 2. 计算预测值和残差
        self.fitted_values_ = X_design @ self.coef_
        self.residuals_ = y - self.fitted_values_

        # 3. 计算 σ̂² (误差方差的无偏估计)
        RSS = np.sum(self.residuals_**2)
        self.df_resid_ = n - p
        self.sigma2_ = RSS / self.df_resid_

        # 4. 计算系数协方差矩阵
        self.cov_matrix_ = self.sigma2_ * XtX_inv

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        生成预测值

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵

        Returns:
        --------
        y_pred : np.ndarray
            预测值 ŷ = Xβ̂
        """
        if self.coef_ is None:
            raise RuntimeError("必须先调用 fit() 训练模型")

        X_design = self._add_intercept(X)
        return X_design @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算 R² (拟合优度)

        R² = 1 - SSE/SST
        SSE = Σ(y_i - ŷ_i)²
        SST = Σ(y_i - ȳ)²

        Returns:
        --------
        r2 : float
            决定系数，取值范围 [0, 1]
        """
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)

        return 1 - SSE / SST

    def f_test(self, C: np.ndarray, d: np.ndarray) -> Dict[str, float]:
        """
        执行一般线性假设检验 H₀: Cβ = d

        Parameters:
        -----------
        C : np.ndarray
            约束矩阵，shape (q, p)
        d : np.ndarray
            约束向量，shape (q,)

        Returns:
        --------
        result : dict
            包含 'f_stat' 和 'p_value'
        """
        if self.coef_ is None:
            raise RuntimeError("必须先调用 fit() 训练模型")

        C = np.asarray(C)
        d = np.asarray(d).reshape(-1, 1)
        q = C.shape[0]  # 约束个数

        # 计算 Cβ̂ - d
        C_beta = C @ self.coef_.reshape(-1, 1)
        diff = C_beta - d

        # 计算 (XᵀX)⁻¹
        XtX = self._X_design.T @ self._X_design
        XtX_inv = np.linalg.inv(XtX)

        # 计算 [C (XᵀX)⁻¹ Cᵀ]⁻¹
        C_XtX_inv_Ct = C @ XtX_inv @ C.T
        C_XtX_inv_Ct_inv = np.linalg.inv(C_XtX_inv_Ct)

        # 计算 F 统计量
        quad_form = diff.T @ C_XtX_inv_Ct_inv @ diff
        f_stat = quad_form.item() / (q * self.sigma2_)

        # 计算 p 值
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)

        return {"f_stat": f_stat, "p_value": p_value}


# ===================================================================
# 过程式实现（仅供对比，不推荐使用）
# ===================================================================


def procedural_fit(X, y, fit_intercept=True):
    """过程式拟合"""
    if fit_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])

    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    residuals = y - X @ beta_hat
    RSS = np.sum(residuals**2)
    df_resid = X.shape[0] - X.shape[1]
    sigma2 = RSS / df_resid
    cov_matrix = sigma2 * XtX_inv

    return beta_hat, cov_matrix, sigma2, df_resid, X


def procedural_predict(X, beta_hat, fit_intercept=True):
    """过程式预测"""
    if fit_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
    return X @ beta_hat


def procedural_score(X, y, beta_hat, fit_intercept=True):
    """过程式计算R²"""
    y_pred = procedural_predict(X, beta_hat, fit_intercept)
    SSE = np.sum((y - y_pred) ** 2)
    SST = np.sum((y - np.mean(y)) ** 2)
    return 1 - SSE / SST


def procedural_f_test(C, d, beta_hat, cov_matrix, sigma2, df_resid, X_design):
    """过程式F检验"""
    C = np.asarray(C)
    d = np.asarray(d).reshape(-1, 1)
    q = C.shape[0]

    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    C_beta = C @ beta_hat.reshape(-1, 1)
    diff = C_beta - d

    C_XtX_inv_Ct = C @ XtX_inv @ C.T
    C_XtX_inv_Ct_inv = np.linalg.inv(C_XtX_inv_Ct)

    quad_form = diff.T @ C_XtX_inv_Ct_inv @ diff
    f_stat = quad_form.item() / (q * sigma2)
    p_value = 1 - stats.f.cdf(f_stat, q, df_resid)

    return {"f_stat": f_stat, "p_value": p_value}
