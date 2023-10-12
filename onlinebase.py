# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:24:46 2021

@author: Wang
"""

import numpy as np
import numpy.matlib
from threading import Lock
class OnlineBase():
    def __init__(self,
                 num_channels: int = 8,
                 sfreq: int = 500,
                 online_whitening: bool = True,
                 blockSize: int = None,
                 forgetfactor: str = "cooling",
                 localstat: np.ndarray = np.inf, 
                 ffdecayrate: float = 0.6, 
                 evalConvergence: bool = True):
        '''
        在线独立成分分解算法(Independent Component Analysis)。
        Parameters
        ----------
        num_channels : int
            电极通道数。
        sfreq: int
            采样频率。
        online_whitening: bool
            是否执行在线矩阵白化。
        blockSize: int
            指定数据块长度。
        forgetfactor: str
            计算遗忘因子的模式。
        localstat: np.ndarray
        ffdecayrate: float
            衰减因子。
        evalConvergence: bool
            收敛性评价。
        '''
        
        #初始化白化
        self._blockSize = blockSize
        self.online_whitening = online_whitening
        self._num_channels = num_channels
        if online_whitening:
            self.W_whitening = np.matrix(np.eye(self._num_channels))
            
        #初始化遗忘因子ForgottingFactor
        self.adaptiveFF_profile = forgetfactor
        self.adaptiveFF_tauconst = np.inf
        #初始化cooling遗忘因子cooling_ff
        self.adaptiveFF_gamma = 0.6
        self.adaptiveFF_lambda_0 = 0.995
        #初始化自适应遗忘因子adaptive_ff
        self.adaptiveFF_decayRateAlpha = 0.02
        self.adaptiveFF_upperBoundBeta = 1e-3
        self.adaptiveFF_transBandWidthGamma = 1
        self.adaptiveFF_transBandCenter = 5
        self.adaptiveFF_lambdaInitial = 0.1
        
        #收敛性评价Non-Stationarity Index (NSI)
        self.evalConverge_profile = evalConvergence
        self.evalConverge_leakyAvgDelta = 1e-2
        self.evalConverge_leakyAvgDeltaVar = 1e-3
        
        #获得眼电通道索引
        self._idx = None
        
        #时间窗
        self._sfreq = sfreq
        
            
    def getIndex(self, idx: int) -> int:
        '''
        获取噪声通道置零索引。

        Parameters
        ----------
        idx : int
            置零通道索引

        -------

        '''
        
        if isinstance(self._idx, int):
            self._idx = idx
            
    def _init_state_structure(self):
        
        #初始化遗忘因子参数 
        self.state_lambda_k = np.zeros((1,self._blockSize))
        self.state_minNonStatIdx = []
        self.state_counter = 0

        if self.adaptiveFF_profile == 'cooling' or self.adaptiveFF_profile =='constant':
            self.adaptiveFF_lambda_const  = 1-np.exp(-1/(self.adaptiveFF_tauconst))
    
        if self.evalConverge_profile:
            self.state_Rn = None
            self.state_nonStatIdx = None
    
        self.state_kurtsign = np.ones((self._num_channels,1)) > 0   
        if self.numSubgaussian != 0:
            self.state.kurtsign[:self.numSubgaussian] = False
            
        
    def _dynamicWhitening(self, block: np.ndarray, data_range: np.ndarray) -> np.ndarray:
        '''
        在线矩阵白化。目的是让个电极之间相关性降低。

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
        W_whitening : np.ndarray, shape(channels,channels)
            解混矩阵，用于分解block。
        W_whitening_inv : np.ndarray, (channels,channels)
            解混矩阵的逆矩阵，lock某一行向量置零后用于还原block。

        '''
        assert block.shape[0] == self._num_channels
        with Lock():
            W_whitening = self.W_whitening
        N,T = block.shape
        if self.adaptiveFF_profile == "cooling":
            lam = self._genCoolingFF(self.state_counter + data_range)
            if lam[0] < self.adaptiveFF_lambda_const:
                lam = np.matlib.repmat(self.adaptiveFF_lambda_const, 1, T)
        elif self.adaptiveFF_profile == "constant":
            lam = np.matlib.repmat(self.adaptiveFF_lambda_const, 1, T)
        elif self.adaptiveFF_profile == "adaptive":
            lam = np.matlib.repmat(self.state_lambda_k[-1], 1, T)
        v = self.W_whitening * block
        lambda_avg = 1 - lam[np.int32(np.ceil(lam[-1] / 2))]
        q_white = lambda_avg / (1 - lambda_avg) + np.trace((v.getH() * v) / T)
        W_whitening = 1/lambda_avg * (W_whitening - v*v.T / T / q_white * W_whitening)
        with Lock():
            self.W_whitening = W_whitening
            self.W_whitening_inv = W_whitening.I
            
        return self.W_whitening
    
    def _genCoolingFF(self, T: np.ndarray) -> float:
        '''
        计算遗忘因子。

        Parameters
        ----------
        T : np.ndarray, shape(times)
            每个数据块的长度，用来计算遗忘因子。数值随自然时间变化而逐渐变大，例如
            在第一个数据块数值是1,2,3,...，后面的数据快将接着第一个数据块的数值继
            续往下数。

        Returns
        -------
        lam: float
            返回遗忘因子。

        '''
        lam = np.divide(self.adaptiveFF_lambda_0, np.power(T, self.adaptiveFF_gamma))
        return lam
  
    def _genAdaptiveFF(self, data_range: np.ndarray, ratio_norm_rn: np.ndarray):
        gain_for_errors = (self.adaptiveFF_upperBoundBeta
                            * 0.5
                            * (1 + np.tanh((ratio_norm_rn - self.adaptiveFF_transBandCenter)
                                / self.adaptiveFF_transBandWidthGamma)))
        f = lambda n : (np.power((1 + gain_for_errors), n)
                                    * self.state_lambda_k[-1]
                                    - self.adaptiveFF_decayRateAlpha
                                    * (np.power((1 + gain_for_errors), (2 * n - 1))
                                        - np.power((1 + gain_for_errors), (n - 1)))
                                    / gain_for_errors
                                    * (self.state_lambda_k[-1] ** 2))
        self.state_lambda_k = f[1:len(data_range)]
            
    def _forgetting_rate(self, T: int, data_range: np.ndarray):
        '''
        使用三种方式得到遗忘因子。
        cooling:
        constant:
        adaptive:

        Parameters
        ----------
        T : int
            数据块长度。
        data_range : np.ndarray, shape(times)
            每个数据块的长度，用来计算遗忘因子。数值随自然时间变化而逐渐变大，例如
            在第一个数据块数值是1,2,3,...，后面的数据快将接着第一个数据块的数值继
            续往下数。
        '''
        # compute the forgetting rate
        if self.adaptiveFF_profile == "cooling":
            self.state_lambda_k = self._genCoolingFF(self.state_counter + data_range)
            if self.state_lambda_k[1] < self.adaptiveFF_lambda_const:
                self.state_lambda_k = np.matlib.repmat(self.adaptiveFF_lambda_const, 1, T)
            self.state_counter = self.state_counter + T
        elif self.adaptiveFF_profile == "constant":
            self.state_lambda_k = np.matlib.repmat(self.adaptiveFF_lambda_const, 1, T)
        elif self.adaptiveFF_profile == "adaptive":
            if not self.state_minNonStatIdx:
                self.state_minNonStatIdx = self.state_nonStatIdx
            self.state_minNonStatIdx = (max(min(self.state_minNonStatIdx,
                                                self.state_nonStatIdx), 1))
            ratio_norm_rn = self.state_nonStatIdx / self.state_minNonStatIdx
            self._genAdaptiveFF(data_range,ratio_norm_rn)
            
    def _NSI(self, N: int, T: int, y: np.ndarray, f: np.ndarray, ):
        '''
        计算不稳定指标。

        Parameters
        ----------
        N : int
            电极通道数。
        T : int
            数据块长度。
        y : np.ndarray
            被解混的信号。
        f : callable
            非线性函数。
        -------

        '''
        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConverge_profile:
            model_fitness = np.identity(N) + y * f.T / T
            #var = np.dot(block, block)
            #print(self.state_Rn)
            if self.state_Rn is None:
                self.state_Rn = model_fitness
            else:
                self.state_Rn = ((1 - self.evalConverge_leakyAvgDeltaVar)
                                    * self.state_Rn
                                    + self.evalConverge_leakyAvgDeltaVar
                                    * model_fitness)
            self.state_nonStatIdx = np.linalg.norm(self.state_Rn, "fro")
            