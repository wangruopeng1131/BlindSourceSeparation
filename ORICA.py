# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:44:08 2021

@author: 王若鹏
"""

import numpy as np
import numpy.matlib
from onlinebase import OnlineBase
from threading import Thread, Lock


class OnlineRecursiveIndependentComponentAnalysis(OnlineBase):
    def __init__(self,
                 num_channels: int = 8,
                 sfreq: int = 400,
                 W_ica: np.ndarray = None,
                 blockSize: int = 32,
                 numSubgaussian: int = 0,
                 nonlinearfunction: callable = None
                 ):
        super().__init__(num_channels=num_channels,
                         blockSize=blockSize,
                         sfreq=sfreq)

        '''
        在线独立成分分解算法(Independent Component Analysis)。
        Parameters
        ----------
        num_channels : int
            电极通道数。
        W_ica : np.ndarray, shape(channels,channels)
            ICA的解混矩阵。若是None，将初始化为单位矩阵。
        blockSize: int
            指定数据块长度。
         numSubgaussian : int
             假定的亚高斯分布数量。
         nonlinearfunction : callable, function
             非线性激活函数。默认为np.tanh。
        '''

        self._num_channels = num_channels
        # 初始化ICA解混矩阵
        if W_ica is None:
            self.W_ica = np.matrix(np.eye(self._num_channels))
        else:
            self.W_ica = np.asmatrix(W_ica)

        self.numSubgaussian = numSubgaussian
        self.nlfunc = nonlinearfunction
        self._count = 0
        self._init_state_structure()

        # 存储原始数据和变换数据，大小为shape(电极数，1)
        self._win = np.zeros((num_channels, 1))
        self._data_to_pipline = np.zeros((num_channels, 1))

    def push(self, block: np.ndarray) -> np.ndarray:
        """
        接收block数据更新解混矩阵W。若有电极置零索引则按以下步骤执行，
        解混->置零->还原。

        Parameters
        ----------
        block : np.ndarray, shape(channels,times)
            每次获取的数据块，时间长度不限。
        Raises
        ------
        ValueError
            block电极数应等于预设定电极数

        Returns
        -------
        block : np.ndarray, shape(channels,times)
            解混的block或者降噪后的block

        """
        N, T = block.shape
        self._win = np.append(self._win, block, axis=-1)
        data_to_pipline = self._data_to_pipline
        if N != self._num_channels:
            raise ValueError('block电极数不等于预设定电极数')
        # 判断win长度是否大于设定blockSize大小
        if self._win.shape[-1] < self._blockSize:
            # 吐出变换后的数据
            if data_to_pipline.shape[-1] > 1:
                self._data_to_pipline = data_to_pipline[:, T:]
                return np.array(data_to_pipline[:, :T])
            else:
                return np.zeros((N, T))
        else:
            win = self._win[:, 1: self._blockSize + 1]
            # 零均值
            win -= win.mean(-1, keepdims=True)
            win = np.matrix(win)
            t = win.shape[-1]
            # 计算自然时间的数据长度
            data_range = np.arange(self._count * t, (self._count + 1) * t) + 1
            self._count += 1
            # 更新白化W
            if self.online_whitening:
                whitethread = Thread(target=self._dynamicWhitening, args=(win, data_range))
                whitethread.start()
                # 白化
                win = self.W_whitening * win
            # 更新ICA W
            icathread = Thread(target=self._dynamicOrica, args=(win, data_range))
            icathread.start()
            # ICA解混
            win = self.W_ica * win

            # 若接收到眼电通道索引则置0
            if self._idx is not None:
                win[self._idx] = 0
                win = self.W_ica_inv * win
                if self.online_whitening:
                    win = self.W_whitening_inv * win

            self._win = self._win[:, self._blockSize + 1:]
            self._data_to_pipline = np.append(data_to_pipline, win[:, T:], axis=-1)
            return np.array(win[:, :T])

    def _dynamicOrica(self, block: np.ndarray, data_range: np.ndarray):
        """
        在线计算解混矩阵W。此方法通过线程调用，目的是不影响主线程的运算速度。

        Parameters
        ----------
        block : np.ndarray, shape(channels,times)
            每次获取的数据块，时间长度不限。
        data_range : np.ndarray, shape(times)
            每个数据块的长度，用来计算遗忘因子。数值随自然时间变化而逐渐变大，例如
            在第一个数据块数值是1,2,3,...，后面的数据快将接着第一个数据块的数值继
            续往下数。

        Returns
        -------
        不返回值，只会不断地更新W_ica,W_ica_inv。
        W_ica : np.ndarray, shape(channels,channels)
            解混矩阵，用于分解block。
        W_ica_inv : np.ndarray, (channels,channels)
            解混矩阵的逆矩阵，lock某一行向量置零后用于还原block。
        """

        N, T = block.shape
        f = np.zeros((N, T))
        with Lock():
            W_ica = self.W_ica
        y = W_ica * block

        self.state_kurtsign = np.multiply(self.state_kurtsign, 1)
        # 非线性函数激活y = w * x的数据 
        if not self.nlfunc:
            f[:] = -2 * np.tanh(np.asarray(y[:]))
            f[self.state_kurtsign, :] = -2 * np.tanh(np.asarray(y[self.state_kurtsign, :]))
        else:
            f = self.nlfunc(y)

        # 计算稳定性指标
        # self._NSI(N,y,f,T)
        # 计算遗忘因子
        self._forgetting_rate(T, data_range)

        # 更新ICA解混矩阵W
        lambda_prod = np.prod(np.divide(1, (1 - self.state_lambda_k)))
        q = 1 + (self.state_lambda_k * np.diagonal(np.inner(f.T, y.T) - 1))
        diag = np.diag(np.divide(self.state_lambda_k, q))
        diag = y * diag
        W_ica = np.multiply(lambda_prod, (W_ica - np.matmul(diag, np.transpose(f)) * W_ica))
        # 正交分解解混矩阵W
        D, V = np.linalg.eig(W_ica * W_ica.T)
        D = np.matrix(np.diag(D))
        V = np.matrix(V)
        W_ica = (V * np.sqrt(D).I * V.T * W_ica)
        with Lock():
            self.W_ica = W_ica
            self.W_ica_inv = W_ica.I


if __name__ == "__main__":
    import mne

    raw_data = mne.io.read_raw_brainvision(r'D:/新建文件夹/5.25/YQP_AP300_2_2021-05-25_11-18-03.vhdr')
    raw_data.load_data()
    raw_data.filter(0.5, 50)
    raw_data.resample(200)
    data_df = raw_data.to_data_frame()
    data_df = data_df.transpose()
    data = data_df.to_numpy()[1:, :]
    c = np.zeros((32, 1))
    orica = OnlineRecursiveIndependentComponentAnalysis(num_channels=32, sfreq=200, blockSize=200)
    for i in range(0, data.shape[-1], 32):
        block = data[:, i:i + 32]
        ica_data = orica.push(block)
        c = np.append(c, ica_data, axis=1)
    c = np.array(c)[:, 1:]
