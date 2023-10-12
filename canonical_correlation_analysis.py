# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:03:35 2021

@author: 王若鹏
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf


class CanonicalCcorrelationAnalysis(object):
    def __init__(self):
        '''
        典型相关性分析实现。特征值分解求解。

        Parameters
        ----------

        Returns
        -------
        None.

        '''

        self._r = None
        self.ccaCoefficient = None
        self.x_bar = None

    def _whiten(self, x: np.ndarray) -> np.ndarray:
        x -= x.mean(-1, keepdims=True)
        A = x @ x.T
        D, V = eigh(A)
        white = V.dot(np.diag(1. / np.sqrt((D + 1e-3)))).dot(V.T)
        x = white.dot(x)
        self.white = white
        self.invWhite = np.linalg.inv(white)
        return x

    def _timedelay(self, x: np.ndarray) -> np.ndarray:
        return x[:, :-1], x[:, 1:]

    def fit(self, X: np.ndarray):
        """
        执行CCA分解。

        Parameters
        ----------
        X : np.ndarray
            需分解矩阵。

        Returns
        -------
        self

        """
        # 矩阵白化
        x = self._whiten(X)
        l = len(X)

        # 数据延迟
        x, y = self._timedelay(X)

        # 算协方差矩阵
        cov = LedoitWolf().fit(np.vstack((x, y)).T).covariance_
        Sxx = cov[:l, :l] + np.eye(l) * 1e-8
        Syy = cov[l:, l:] + np.eye(l) * 1e-8
        Sxy = cov[l:, :l]
        Syx = cov[:l, l:]
        invSyy = np.linalg.inv(Syy)
        # 典型相关性分析求解，特征分解方法
        r, Wx = np.linalg.eig(np.linalg.inv(Sxx).dot(Sxy).dot(invSyy).dot(Syx))
        r = np.sqrt(np.real(r + 1e-3))
        # 降序重排
        sort = np.real(r).argsort()[::-1]
        r = r[sort]
        Wx = Wx[:, sort]
        Wx = np.real(Wx)
        self.ccaCoefficient = Wx.T
        self._r = r
        return self

    def fit_transform(self, x: np.array) -> np.array:
        self.fit(x)
        return self.transform(x)

    def transform(self, x: np.array) -> np.array:
        x_decompose = self.ccaCoefficient.dot(x)
        return x_decompose

    def inverse_transform(self, x: np.array) -> np.array:
        result = np.linalg.inv(self.ccaCoefficient).dot(x)
        # result = self.invWhite.dot(result)
        return result

    @property
    def get_auto_correlation(self):
        return self._r

    @property
    def get_singular_vectors(self):
        return self.ccaCoefficient

    @property
    def get_whiten_matrix(self):
        return self.white

    @property
    def get_inv_whiten_matrix(self):
        return self.invWhite
