# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:38:54 2021

@author: 王若鹏
"""
import numpy as np
from scipy.linalg import sqrtm, qr, qr_update, eigh
from sklearn.decomposition import PCA
from copy import deepcopy


def _vecnorm(vec: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(vec)
    vec = vec / mag
    return vec.squeeze()


class IndependentVectorAnalysis(object):
    def __init__(self, n_components: float,
                 maxIter: int = 500,
                 WDiffStop: float = 1e-6,
                 alphaMin: float = 1e-6,
                 alpha0: float = 1.,
                 blowup: float = 1e3,
                 alphaScale: float = 0.9):
        """
        独立分量分析实现（IVA）。

        Parameters
        ----------
        n_components : int
            独立成分数量
        maxIter : int, optional
            最大迭代次数. The default is 500.
        WDiffStop : float, optional
            解混矩阵W收敛值. The default is 1e-6.
        alphaMin : float, optional
            DESCRIPTION. The default is 1e-6.
        alpha0 : float, optional
            DESCRIPTION. The default is 1..
        blowup : float, optional
            DESCRIPTION. The default is 1e3.
        alphaScale : float, optional
            DESCRIPTION. The default is 0.9.

        Returns
        -------
        None.

        """

        self.x_whiten = None
        self.detSCV = None
        self.W_inv = None
        self.W = None
        self.x_double = None
        self.x_pca = None
        self.maxIter = maxIter
        self.WDiffStop = WDiffStop
        self.alphaMin = alphaMin
        self.alpha0 = alpha0
        self.blowup = blowup
        self.alphaScale = alphaScale
        self.pca = PCA(n_components, svd_solver='full')

    def fit(self, x: np.ndarray):

        assert len(x.shape) == 2

        x -= x.mean(-1, keepdims=True)
        x = self._whitening(x)

        # 降维
        x = self.pca.fit_transform(x.T).T
        self.x_pca = x
        self.W = np.random.rand(2, x.shape[0], x.shape[0])
        x = np.array([x[:, 1:], x[:, :-1]]).transpose(1, 2, 0)
        self.x_double = x
        alpha0 = self.alpha0
        N, T, K = x.shape
        W, Rx = self._init(x)

        cost = 0
        cost_const = K * np.log(2 * np.pi * np.exp(1))
        grad = np.zeros((K, N))
        # 循环开始
        for iTer in range(self.maxIter):
            termCriterion = 0
            W_old = deepcopy(W)
            cost_old = deepcopy(cost)
            cost -= np.log(np.abs(np.linalg.det(W))).sum()

            Q, R = 0., 0.
            for n in range(N):
                Sigma_n = np.eye(K)
                for k1 in range(K):
                    for k2 in range(K):
                        temp1 = W[k1, n, :]
                        temp2 = W[k2, n, :]
                        Sigma_n[k1, k2] = temp1.dot(Rx[k1][k2]).dot(temp2.T)
                        if k1 != k2:
                            Sigma_n[k2, k1] = Sigma_n[k1, k2].T

                cost -= 0.5 * (cost_const + np.log(np.linalg.det(Sigma_n)))
                inv_Sigma_n = np.linalg.inv(Sigma_n)

                # decouple_trick
                hnk, Q, R = self._decouple_trick(K, N, W, n, Q, R)

                for k in range(K):
                    grad[k] = -hnk[k] / W[k, n, :].dot(hnk[k])
                    for kk in range(K):
                        grad[k] += Rx[k][kk].dot(W[kk, n, :].T).dot(inv_Sigma_n[kk, k])
                    wnk = W[k, n, :]
                    gradNorm = _vecnorm(grad[k])
                    gradNormProj = _vecnorm(gradNorm - wnk.T.dot(gradNorm) * wnk)
                    W[k, n, :] = _vecnorm(wnk - alpha0 * gradNormProj)
                    for kk in range(K):
                        Sigma_n[k, kk] = W[k, n, :].dot(Rx[k][kk]).dot(W[kk, n, :].T)
                    Sigma_n[k] = Sigma_n[:, k].T
                    inv_Sigma_n = np.linalg.inv(Sigma_n)

            for k in range(K):
                termCriterion = max(termCriterion, max(1 - np.abs(np.diag(W_old[k].dot(W[k].T)))))
            if iTer > 1:
                if cost > cost_old:
                    alpha0 = max(self.alphaMin, self.alphaScale * alpha0)

            if termCriterion < self.WDiffStop or iTer == self.maxIter:
                break
            elif termCriterion > self.blowup or np.isnan(cost):
                for k in K:
                    W[k] = np.eye(N) + 0.1 * np.random.rand(N)
        # 循环结束

        # 矩阵单位化
        for n in range(N):
            for k in range(K):
                W[k, n, :] = W[k, n, :] / np.sqrt(W[k, n, :].dot(Rx[k][k]).dot(W[k, n, :].T))

        # SCVs重排序
        # Order the components from most to least ill-conditioned

        # Second, compute the determinant of the SCVs
        detSCV = np.zeros((N))
        Sigma_N = np.zeros((K, K, N))
        for n in range(N):
            # Efficient version of Sigma_n=Yn*Yn'/T;
            Sigma_n = np.zeros((K, K))
            for k1 in range(K):
                for k2 in range(k1, K):
                    Sigma_n[k1, k2] = W[k1, n, :].dot(Rx[k][k2]).dot(W[k2, n, :].T)
                    Sigma_n[k2, k1] = Sigma_n[k1, k2].T
            Sigma_N[..., n] = Sigma_n

            detSCV[n] = np.linalg.det(Sigma_n)

        # Third, sort and apply
        sort = detSCV.argsort()[::-1]
        Sigma_N = Sigma_N[:, :, sort]
        for k in range(K):
            W[k, :, :] = W[k, sort, :]
        self.detSCV = detSCV
        self.W = W
        self.W_inv = np.linalg.inv(W)
        return self

    def fit_laplace(self):

        alpha0 = 0.1
        cost = 0
        x = self.x_double.transpose(2, 0, 1)
        W = self.W
        K, N, T = x.shape
        for iTer in range(self.maxIter):
            termCriterion = 0

            # 计算SCVs
            Y = np.matmul(W, x)

            # 计算SCVs间的内积
            innerY = np.linalg.norm(Y, axis=0)

            # 计算cost
            cost_old = deepcopy(cost)
            cost += np.log(np.abs(np.linalg.det(W))).sum()
            cost = innerY.sum() / (T - cost)
            cost /= N * K

            W_old = deepcopy(W)
            Q, R = 0, 0
            for n in range(N):
                # decouple_trick
                hnk, Q, R = self._decouple_trick(K, N, W, n, Q, R)

                for k in range(K):
                    phi = (Y[k, n, :] / np.linalg.norm(Y[:, n, :], axis=0)).reshape(1, -1)

                    dW = ((K + 1) / 2 * x[k].dot(phi.T) / T).squeeze() - hnk[k] / W[k, n, :].dot(hnk[k])
                    W[k, n, :] = _vecnorm(W[k, n, :] - alpha0 * dW)
                    Y[k, n, :] = W[k, n, :].dot(x[k])
            # Y = np.matmul(W,x)

            termCriterion = max(termCriterion, max((1 - np.abs(np.diag(np.matmul(W_old[k, :, :], W[k, :, :]), 1)))))

            if termCriterion < self.WDiffStop or iTer == self.maxIter:
                break
            elif np.isnan(cost):
                for k in range(0, K):
                    W[k, :, :] = np.eye(N) + 0.1 * np.random.randn(N)
            elif iTer > 1 and cost > cost_old:
                alpha0 = max(self.alphaMin, self.alphaScale * alpha0)

        self.W = W
        self.W_inv = np.linalg.inv(W)
        return self

    def transform(self, x):
        w = self.W[0]
        result = w.dot(x)
        return result

    def fit_transform(self, x):
        self.fit(x)
        self.fit_laplace()
        result = self.transform(self.x_pca)
        return result

    def inverse_transform(self, x):
        w_inv = self.W_inv[0]
        result = w_inv.dot(x)
        result = self.pca.inverse_transform(result.T).T
        result = np.linalg.inv(self.whitenMatrix).dot(result)
        return result

    def _init(self, x):
        N, T, K = x.shape
        W = self.W
        for k in range(K):
            W[k] = np.linalg.inv(W[k]).dot(sqrtm(W[k].dot(W[k].T)))

        Rx = [[[] for _ in range(K)] for _ in range(K)]
        for k1 in range(K):
            Rx[k1][k1] = x[..., k1].dot(x[..., k1].T) / T
            for k2 in range(k1 + 1, K):
                Rx[k1][k2] = x[..., k1].dot(x[..., k2].T) / T
                Rx[k2][k1] = Rx[k1][k2].T

        return W, Rx

    def _decouple_trick(self, K, N, W, n, Q, R):
        '''
        cite: Nonorthogonal Joint Diagonalization Free of Degenerate Solution
        IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 55, NO. 5, MAY 2007
        '''
        h = np.zeros((K, N))
        if n == 0:
            invQ = np.zeros((K, N, N))
            R = np.zeros((K, N, N - 1))
        else:
            invQ = Q
        for k in range(K):
            if n == 0:
                Wtilde = W[k, 1:N, :]
                invQ[k], R[k] = qr(Wtilde.T)
            else:
                n_last = n - 1
                e_last = np.zeros((N - 1, 1))
                e_last[n_last] = 1
                u = -W[k, n, ...].reshape(-1, 1)
                invQ[k], R[k] = qr_update(invQ[k], R[k], u, e_last, True)
                u = W[k, n_last, :].reshape(-1, 1)
                invQ[k], R[k] = qr_update(invQ[k], R[k], u, e_last, True)
            h[k, :] = invQ[k, :, -1]
        return h, invQ, R

    def _whitening(self, x: np.ndarray) -> np.ndarray:
        A = x.dot(x.T)
        D, V = eigh(A)
        white = V.dot(np.diag(1. / np.sqrt((D + 1e-3)))).dot(V.T)
        x = white.dot(x)
        self.whitenMatrix = white
        return x

    @property
    def get_demix_matrix(self):
        return self.W

    @property
    def get_mix_matrix(self):
        return self.W_inv

    @property
    def get_auto_correlation(self):
        return self.detSCV

    @property
    def get_whiten_matrix(self):
        return self.whitenMatrix

    @property
    def get_inv_whiten_matrix(self):
        return np.linalg.inv(self.whitenMatrix)

    @property
    def get_whiten_input(self):
        return self.x_whiten

    @property
    def get_double_input(self):
        return self.x_double.transpose(2, 0, 1)


# TODO: IVA-GL
if __name__ == "__main__":
    from scipy import io
    from statsmodels.tsa.stattools import acf
    from plot import plot_imf
    X = io.loadmat(r'F:/project/denoise/ReMAE/data/multichanneldata.mat')['multichanneldata']
    # plot_imf(X, name="Raw.png", save=True)
    iva = IndependentVectorAnalysis(21)
    # iva.fit(X)
    y = iva.fit_transform(X)
    # plot_imf(y, name="Independent Vector.png", save=True)
    ac = np.array([acf(yy)[1] for yy in y])
    where = np.where(ac < 0.9)
    y[where] = 0
    x_r = iva.transform(y)
    # plot_imf(x_r, name="Reconstruct.png", save=True)