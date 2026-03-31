import torch
from .fundamental_abc import Fundamental


class GaussianMeanReverting(Fundamental):
    '''
    Docstring for GaussianMeanReverting
    '''
    def __init__(self, final_time: int, mean: float, r: float, shock_var: float, shock_mean: float = 0):
        '''
        Docstring for __init__
        
        :param self: Description
        :param final_time: Description
        :type final_time: int
        :param mean: Description
        :type mean: float
        :param r: Description
        :type r: float
        :param shock_var: Description
        :type shock_var: float
        :param shock_mean: Description
        :type shock_mean: float
        '''
        self.final_time = final_time
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.r = torch.tensor(r, dtype=torch.float32)
        self.shock_mean = shock_mean
        self.shock_std = torch.sqrt(torch.tensor(shock_var, dtype=torch.float32))
        self.fundamental_values = torch.zeros(final_time, dtype=torch.float32)
        self.fundamental_values[0] = mean
        self._generate()

    def _generate(self):
        '''
        Docstring for _generate
        
        :param self: Description
        '''
        shocks = torch.randn(self.final_time)*self.shock_std + self.shock_mean
        for t in range(1, self.final_time):
            self.fundamental_values[t] = (
                max(0, self.r*self.mean + (1 - self.r)*self.fundamental_values[t - 1] + shocks[t])
            )

    def get_value_at(self, time: int) -> float:
        '''
        Docstring for get_value_at
        
        :param self: Description
        :param time: Description
        :type time: int
        :return: Description
        :rtype: float
        '''
        return self.fundamental_values[time].item()

    def get_fundamental_values(self) -> torch.Tensor:
        '''
        Docstring for get_fundamental_values
        
        :param self: Description
        :return: Description
        :rtype: Tensor
        '''
        return self.fundamental_values

    def get_final_fundamental(self) -> float:
        '''
        Docstring for get_final_fundamental
        
        :param self: Description
        :return: Description
        :rtype: float
        '''
        return self.fundamental_values[-1].item()

    def get_r(self) -> float:
        '''
        Docstring for get_r
        
        :param self: Description
        :return: Description
        :rtype: float
        '''
        return self.r.item()

    def get_mean(self) -> float:
        '''
        Docstring for get_mean
        
        :param self: Description
        :return: Description
        :rtype: float
        '''
        return self.mean.item()

    def get_info(self):
        '''
        Docstring for get_info
        
        :param self: Description
        '''
        return self.get_mean(), self.get_r(), self.final_time
