from typing import Dict, Tuple, Union

from scipy.signal import butter, lfilter
from scipy.signal.windows import hann

import numpy as np

# https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/transforms_interface.py

from typing import Callable, Dict, Union, List

from typing import Callable, List



class BaseTransform:
    def __init__(self):
        self._additional_targets: Dict[str, str] = {}

    def __call__(self, *args, **kwargs) -> Dict[str, any]:
        if args:
            raise KeyError("Please pass data as named parameters.")
        res = {}

        params = self.get_params()

        if self.targets_as_params:
            assert all(key in kwargs
                       for key in self.targets_as_params), "{} requires {}".format(self.__class__.__name__,
                                                                                   self.targets_as_params)
            targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
            params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)

        for key, arg in kwargs.items():
            if not arg is None:
                target_function = self._get_target_function(key)
                res[key] = target_function(arg, **params)
        return res

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params(self) -> Dict:
        return {}

    def get_params_dependent_on_targets(self, params: Dict[str, any]) -> Dict[str, any]:
        return {}

    def _get_target_function(self, key: str) -> Callable:
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def add_targets(self, additional_targets: Dict[str, str]):
        self._additional_targets = additional_targets

    @property
    def targets(self) -> Dict[str, Callable]:
        raise NotImplementedError("Method targets is not implemented in class " + self.__class__.__name__)

    def apply(self, *args, **kwargs) -> any:
        raise NotImplementedError("Method apply is not implemented in class " + self.__class__.__name__)

    @property
    def repr_body(self) -> Dict:
        return {}

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ', '
            # str param
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ')'
        return format_string


class Compose(BaseTransform):
    r'''
    Compose several transforms together. Consistent with :obj:`torchvision.transforms.Compose`'s behavior.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(64, 64)),
            transforms.RandomNoise(p=0.1),
            transforms.RandomMask(p=0.1)
        ])
        t(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 64, 64)

    :obj:`Compose` supports transformers with different data dependencies. The above example combines multiple torch-based transformers, the following example shows a sequence of numpy-based transformer.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Compose([
            transforms.BandDifferentialEntropy(),
            transforms.MeanStdNormalize(),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ])
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        transforms (list): The list of transforms to compose.

    .. automethod:: __call__
    '''
    def __init__(self, transforms: List[Callable]):
        # super(Compose, self).__init__()
        super().__init__()
        self.transforms = transforms

    def __call__(self, *args, **kwargs) -> any:
        r'''
        Args:
            x (any): The input.

        Returns:
            any: The transformed output.
        '''
        if args:
            raise KeyError("Please pass data as named parameters.")

        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ','
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

class EEGTransform(BaseTransform):
    def __init__(self, apply_to_baseline: bool = False):
        # super(EEGTransform, self).__init__()
        super().__init__()
        self.apply_to_baseline = apply_to_baseline
        if apply_to_baseline:
            self.add_targets({'baseline': 'eeg'})

    @property
    def targets(self):
        return {"eeg": self.apply}

    def apply(self, eeg: any, baseline: Union[any, None] = None, **kwargs) -> any:
        raise NotImplementedError("Method apply is not implemented in class " + self.__class__.__name__)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'apply_to_baseline': self.apply_to_baseline})


class LabelTransform(BaseTransform):
    @property
    def targets(self):
        return {"y": self.apply}

    def apply(self, y: any, **kwargs) -> any:
        raise NotImplementedError("Method apply is not implemented in class " + self.__class__.__name__)


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


class BandTransform(EEGTransform):
    def __init__(self,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandTransform, self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in eeg:
                b, a = butter_bandpass(low,
                                       high,
                                       fs=self.sampling_rate,
                                       order=self.order)
                c_list.append(self.opt(lfilter(b, a, c)))
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=-1)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'order': self.order,
                'band_dict': {...}
            })


class BandSignal(BandTransform):
    r'''
    A transform method to split the EEG signal into signals in different sub-bands.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BandSignal()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (4, 32, 128)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The differential entropy of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in eeg:
                b, a = butter(self.order, [low, high],
                              fs=self.sampling_rate,
                              btype="band")
                c_list.append(self.opt(lfilter(b, a, c)))
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=0)


class BandDifferentialEntropy(BandTransform):
    r'''
    A transform method for calculating the differential entropy of EEG signals in several sub-bands with EEG signals as input. It is a widely accepted differential entropy calculation method by the community, which is often applied to the DEAP and DREAMER datasets. It is relatively easy to understand and has a smaller scale and more gradual changes than the :obj:`BandDifferentialEntropyV1` calculated based on average power spectral density.

    - Related Paper: Fdez J, Guttenberg N, Witkowski O, et al. Cross-subject EEG-based emotion recognition through neural networks with stratified normalization[J]. Frontiers in neuroscience, 2021, 15: 626277.
    - Related Project: https://github.com/javiferfer/cross-subject-eeg-emotion-recognition-through-nn/

    - Related Paper: Li D, Xie L, Chai B, et al. Spatial-frequency convolutional self-attention network for EEG emotion recognition[J]. Applied Soft Computing, 2022, 122: 108740.
    - Related Project: https://github.com/qeebeast7/SFCSAN/

    In most cases, choosing :obj:`BandDifferentialEntropy` and :obj:`BandDifferentialEntropyV1` does not make much difference. If you have other comments, please feel free to pull request.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BandDifferentialEntropy()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The differential entropy of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return 1 / 2 * np.log2(2 * np.pi * np.e * np.var(eeg))


class BandDifferentialEntropyV1(EEGTransform):
    r'''
    A transform method for calculating the differential entropy of EEG signals in several sub-bands with EEG signals as input. This version calculates the differential entropy based on the relationship between the differential entropy and the average power spectral density, which is identical to the processing of the SEED dataset.

    - Related Paper: Shi L C, Jiao Y Y, Lu B L. Differential entropy feature for EEG-based vigilance estimation[C]//2013 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2013: 6627-6630.
    - Related Project: https://github.com/ziyujia/Signal-feature-extraction_DE-and-PSD

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BandDifferentialEntropyV1()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __init__(self,
                 sampling_rate: int = 128,
                 fft_n: int = None,
                 num_window: int = 1,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandDifferentialEntropyV1,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate

        if fft_n is None:
            fft_n = self.sampling_rate

        self.fft_n = fft_n
        self.num_window = num_window
        self.band_dict = band_dict

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        _, chunk_size = eeg.shape
        point_per_window = int(chunk_size // self.num_window)

        band_list = []

        for window_index in range(self.num_window):
            start_index, end_index = point_per_window * window_index, point_per_window * (
                    window_index + 1)
            window_data = eeg[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=self.fft_n)
            energy_graph = np.abs(fft_data[:, 0:int(self.fft_n / 2)])

            for _, band in enumerate(self.band_dict.values()):
                start_index = int(
                    np.floor(band[0] / self.sampling_rate * self.fft_n))
                end_index = int(
                    np.floor(band[1] / self.sampling_rate * self.fft_n))
                band_ave_psd = np.mean(energy_graph[:, start_index -
                                                       1:end_index] ** 2,
                                       axis=1)
                # please refer to # https://github.com/ziyujia/Signal-feature-extraction_DE-and-PSD/blob/master/DE_PSD.py, which consider the relationship between DE and PSD to calculate DE.
                band_de = np.log2(100 * band_ave_psd)
                band_list.append(band_de)

        return np.stack(band_list, axis=-1)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The power spectral density of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'fft_n': self.fft_n,
                'num_window': self.num_window,
                'band_dict': {...}
            })


class BandPowerSpectralDensity(EEGTransform):
    r'''
    A transform method for calculating the power spectral density of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BandPowerSpectralDensity()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The sampling rate of EEG signals in Hz. (default: :obj:`128`)
        fft_n (int): Computes the one-dimensional n-point discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT) algorithm. If set to None, it will automatically match sampling_rate. (default: :obj:`None`)
        num_window (int): Welch's method computes an estimate of the power spectral density by dividing the data into non-overlapping segments, where the num_window denotes the number of windows. (default: :obj:`1`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the power spectral density of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __init__(self,
                 sampling_rate: int = 128,
                 fft_n: int = None,
                 num_window: int = 1,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandPowerSpectralDensity,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate

        if fft_n is None:
            fft_n = self.sampling_rate

        self.fft_n = fft_n
        self.num_window = num_window
        self.band_dict = band_dict

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        _, chunk_size = eeg.shape
        point_per_window = int(chunk_size // self.num_window)

        band_list = []

        for window_index in range(self.num_window):
            start_index, end_index = point_per_window * window_index, point_per_window * (
                    window_index + 1)
            window_data = eeg[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=self.fft_n)
            energy_graph = np.abs(fft_data[:, 0:int(self.fft_n / 2)])

            for _, band in enumerate(self.band_dict.values()):
                start_index = int(
                    np.floor(band[0] / self.sampling_rate * self.fft_n))
                end_index = int(
                    np.floor(band[1] / self.sampling_rate * self.fft_n))
                band_ave_psd = np.mean(energy_graph[:, start_index -
                                                       1:end_index] ** 2,
                                       axis=1)

                band_list.append(band_ave_psd)

        return np.stack(band_list, axis=-1)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The power spectral density of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'fft_n': self.fft_n,
                'num_window': self.num_window,
                'band_dict': {...}
            })


class BandMeanAbsoluteDeviation(BandTransform):
    r'''
    A transform method for calculating the mean absolute deviation of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BandMeanAbsoluteDeviation()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the mean absolute deviation of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)

    .. automethod:: __call__
    '''

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The mean absolute deviation of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return np.mean(np.abs(eeg - np.mean(eeg)))


class BandKurtosis(BandTransform):
    r'''
    A transform method for calculating the kurtosis of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BandKurtosis()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the kurtosis of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)

    .. automethod:: __call__
    '''

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The kurtosis of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        n = len(eeg)
        ave1 = 0.0
        ave2 = 0.0
        ave4 = 0.0
        for eeg in eeg:
            ave1 += eeg
            ave2 += eeg ** 2
            ave4 += eeg ** 4
        ave1 /= n
        ave2 /= n
        ave4 /= n
        sigma = np.sqrt(ave2 - ave1 ** 2)
        return ave4 / (sigma ** 4)


class BandSkewness(BandTransform):
    r'''
    A transform method for calculating the skewness of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BandSkewness()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the skewness of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)

    .. automethod:: __call__
    '''

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The skewness of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        n = len(eeg)
        ave1 = 0.0
        ave2 = 0.0
        ave3 = 0.0
        for eeg in eeg:
            ave1 += eeg
            ave2 += eeg ** 2
            ave3 += eeg ** 3
        ave1 /= n
        ave2 /= n
        ave3 /= n
        sigma = np.sqrt(ave2 - ave1 ** 2)
        return (ave3 - 3 * ave1 * sigma ** 2 - ave1 ** 3) / (sigma ** 3)


from typing import Dict, Tuple, Union

from scipy.interpolate import griddata

import numpy as np

class To2d(EEGTransform):
    r'''
    Taking the electrode index as the row index and the temporal index as the column index, a two-dimensional EEG signal representation with the size of [number of electrodes, number of data points] is formed. While PyTorch performs convolution on the 2d tensor, an additional channel dimension is required, thus we append an additional dimension.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.To2d()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (1, 32, 128)

    .. automethod:: __call__
    '''

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The transformed results with the shape of [1, number of electrodes, number of data points].
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg[np.newaxis, ...]


class ToGrid(EEGTransform):
    r'''
    A transform method to project the EEG signals of different channels onto the grid according to the electrode positions to form a 3D EEG signal representation with the size of [number of data points, width of grid, height of grid]. For the electrode position information, please refer to constants grouped by dataset:

    - datasets.constants.emotion_recognition.deap.DEAP_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.dreamer.DREAMER_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.seed.SEED_CHANNEL_LOCATION_DICT
    - ...

    .. code-block:: python

        from torcheeg import transforms
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

        t = transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        channel_location_dict (dict): Electrode location information. Represented in dictionary form, where :obj:`key` corresponds to the electrode name and :obj:`value` corresponds to the row index and column index of the electrode on the grid.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    .. automethod:: reverse
    '''

    def __init__(self,
                 channel_location_dict: Dict[str, Tuple[int, int]],
                 apply_to_baseline: bool = False):
        super(ToGrid, self).__init__(apply_to_baseline=apply_to_baseline)
        self.channel_location_dict = channel_location_dict

        loc_x_list = []
        loc_y_list = []
        for _, locs in channel_location_dict.items():
            if locs is None:
                continue
            (loc_y, loc_x) = locs
            loc_x_list.append(loc_x)
            loc_y_list.append(loc_y)

        self.width = max(loc_x_list) + 1
        self.height = max(loc_y_list) + 1

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The projected results with the shape of [number of data points, width of grid, height of grid].
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        # num_electrodes x timestep
        outputs = np.zeros([self.height, self.width, eeg.shape[-1]])
        # 9 x 9 x timestep
        for i, locs in enumerate(self.channel_location_dict.values()):
            if locs is None:
                continue
            (loc_y, loc_x) = locs
            outputs[loc_y][loc_x] = eeg[i]

        outputs = outputs.transpose(2, 0, 1)
        # timestep x 9 x 9
        return outputs

    def reverse(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        r'''
        The inverse operation of the converter is used to take out the electrodes on the grid and arrange them in the original order.
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of data points, width of grid, height of grid].

        Returns:
            np.ndarray: The revered results with the shape of [number of electrodes, number of data points].
        '''
        # timestep x 9 x 9
        eeg = eeg.transpose(1, 2, 0)
        # 9 x 9 x timestep
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[2]])
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]
        # num_electrodes x timestep
        return {
            'eeg': outputs
        }

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'channel_location_dict': {...}})


class ToInterpolatedGrid(EEGTransform):
    r'''
    A transform method to project the EEG signals of different channels onto the grid according to the electrode positions to form a 3D EEG signal representation with the size of [number of data points, width of grid, height of grid]. For the electrode position information, please refer to constants grouped by dataset:

    - datasets.constants.emotion_recognition.deap.DEAP_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.dreamer.DREAMER_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.seed.SEED_CHANNEL_LOCATION_DICT
    - ...

    .. code-block:: python

        from torcheeg import transforms
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

        t = ToInterpolatedGrid(DEAP_CHANNEL_LOCATION_DICT)
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (128, 9, 9)

    Especially, missing values on the grid are supplemented using cubic interpolation

    Args:
        channel_location_dict (dict): Electrode location information. Represented in dictionary form, where :obj:`key` corresponds to the electrode name and :obj:`value` corresponds to the row index and column index of the electrode on the grid.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    .. automethod:: reverse
    '''

    def __init__(self,
                 channel_location_dict: Dict[str, Tuple[int, int]],
                 apply_to_baseline: bool = False):
        super(ToInterpolatedGrid,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.channel_location_dict = channel_location_dict
        self.location_array = np.array(list(channel_location_dict.values()))

        loc_x_list = []
        loc_y_list = []
        for _, (loc_y, loc_x) in channel_location_dict.items():
            loc_x_list.append(loc_x)
            loc_y_list.append(loc_y)

        self.width = max(loc_x_list) + 1
        self.height = max(loc_y_list) + 1

        grid_y, grid_x = np.mgrid[
                         min(self.location_array[:, 0]):max(self.location_array[:, 0]
                                                            ):self.height * 1j,
                         min(self.location_array[:,
                             1]):max(self.location_array[:,
                                     1]):self.width *
                                         1j, ]
        self.grid_y = grid_y
        self.grid_x = grid_x

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The projected results with the shape of [number of data points, width of grid, height of grid].
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        # channel eeg timestep
        eeg = eeg.transpose(1, 0)
        # timestep eeg channel
        outputs = []

        for timestep_split_y in eeg:
            outputs.append(
                griddata(self.location_array,
                         timestep_split_y, (self.grid_x, self.grid_y),
                         method='cubic',
                         fill_value=0))
        outputs = np.array(outputs)
        return outputs

    def reverse(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        r'''
        The inverse operation of the converter is used to take out the electrodes on the grid and arrange them in the original order.
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of data points, width of grid, height of grid].

        Returns:
            np.ndarray: The revered results with the shape of [number of electrodes, number of data points].
        '''
        # timestep x 9 x 9
        eeg = eeg.transpose(1, 2, 0)
        # 9 x 9 x timestep
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[2]])
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]
        # num_electrodes x timestep
        return {
            'eeg': outputs
        }

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'channel_location_dict': {...}})

# from typing import Dict, Union
#
# import numpy as np

import torch

class ToTensor(EEGTransform):
    r'''
    Convert a :obj:`numpy.ndarray` to tensor. Different from :obj:`torchvision`, tensors are returned without scaling.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.ToTensor()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        apply_to_baseline (bool): Whether to apply the transform to the baseline signal. (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __init__(self, apply_to_baseline: bool = False):
        # super(ToTensor, self).__init__(apply_to_baseline=apply_to_baseline)
        super().__init__(apply_to_baseline=apply_to_baseline)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals.
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            dict: If baseline is passed and apply_to_baseline is set to True, then {'eeg': ..., 'baseline': ...}, else {'eeg': ...}. The output is represented by :obj:`torch.Tensor`.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> torch.Tensor:
        return torch.from_numpy(eeg).float()


from typing import Dict, List, Union

class Select(LabelTransform):
    r'''
    Select part of the value from the information dictionary.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Select(key='valence')
        t(y={'valence': 4.5, 'arousal': 5.5, 'subject_id': 7})['y']
        >>> 4.5

    :obj:`Select` allows multiple values to be selected and returned as a list. Suitable for multi-classification tasks or multi-task learning.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Select(key=['valence', 'arousal'])
        t(y={'valence': 4.5, 'arousal': 5.5, 'subject_id': 7})['y']
        >>> [4.5, 5.5]

    Args:
        key (str or list): The selected key can be a key string or a list of keys.

    .. automethod:: __call__
    '''

    def __init__(self, key: Union[str, List]):
        # super(Select, self).__init__()
        super().__init__()
        self.key = key
        self.select_list = isinstance(key, list) or isinstance(key, tuple)

    def __call__(self, *args, y: Dict, **kwargs) -> Union[int, float, List]:
        r'''
        Args:
            y (dict): A dictionary describing the EEG signal samples, usually as the last return value for each sample in :obj:`Dataset`.

        Returns:
            str or list: Selected value or selected value list.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Dict, **kwargs) -> Union[int, float, List]:
        assert isinstance(
            y, dict
        ), f'The transform Select only accepts label dict as input, but obtain {type(y)} as input.'
        if self.select_list:
            return [y[k] for k in self.key]
        return y[self.key]

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'key': self.key
        })


class Binary(LabelTransform):
    r'''
    Binarize the label according to a certain threshold. Labels larger than the threshold are set to 1, and labels smaller than the threshold are set to 0.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Binary(threshold=5.0)
        t(y=4.5)['y']
        >>> 0

    :obj:`Binary` allows simultaneous binarization using the same threshold for multiple labels.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Binary(threshold=5.0)
        t(y=[4.5, 5.5])['y']
        >>> [0, 1]

    Args:
        threshold (float): Threshold used during binarization.

    .. automethod:: __call__
    '''

    def __init__(self, threshold: float):
        # super(Binary, self).__init__()
        super().__init__()
        self.threshold = threshold

    def __call__(self, *args, y: Union[int, float, List],
                 **kwargs) -> Union[int, List]:
        r'''
        Args:
            label (int, float, or list): The input label or list of labels.

        Returns:
            int, float, or list: The output label or list of labels after binarization.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Union[int, float, List], **kwargs) -> Union[int, List]:
        if isinstance(y, list):
            return [int(l >= self.threshold) for l in y]
        return int(y >= self.threshold)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'threshold': self.threshold})


class BinaryOneVSRest(LabelTransform):
    r'''
    Binarize the label following the fashion of the one-vs-rest strategy. When label is the specified positive category label, the label is set to 1, when the label is any other category label, the label is set to 0.

    .. code-block:: python

        from torcheeg import transforms

        t = BinaryOneVSRest(positive=1)
        t(y=2)['y']
        >>> 0

    :obj:`Binary` allows simultaneous binarization using the same threshold for multiple labels.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BinaryOneVSRest(positive=1)
        t(y=[1, 2])['y']
        >>> [1, 0]

    Args:
        positive (int): The specified positive category label.

    .. automethod:: __call__
    '''

    def __init__(self, positive: int):
        super(BinaryOneVSRest, self).__init__()
        self.positive = positive

    def apply(self, y: Union[int, float, List], **kwargs) -> Union[int, List]:
        assert isinstance(
            y, (int, float, list)
        ), f'The transform Binary only accepts label list or item (int or float) as input, but obtain {type(y)} as input.'
        if isinstance(y, list):
            return [int(l == self.positive) for l in y]
        return int(y == self.positive)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'positive': self.positive})


class BinariesToCategory(LabelTransform):
    r'''
    Convert multiple binary labels into one multiclass label. Multiclass labels represent permutations of binary labels. Commonly used to combine two binary classification tasks into a single quad classification task.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BinariesToCategory()
        t(y=[0, 0])['y']
        >>> 0
        t(y=[0, 1])['y']
        >>> 1
        t(y=[1, 0])['y']
        >>> 2
        t(y=[1, 1])['y']
        >>> 3

    .. automethod:: __call__
    '''

    def __call__(self, *args, y: List, **kwargs) -> int:
        r'''
        Args:
            y (list): list of binary labels.

        Returns:
            int: The converted multiclass label.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: List, **kwargs) -> int:
        r'''
        Args:
            y (list): list of binary labels.

        Returns:
            int: The converted multiclass label.
        '''
        assert isinstance(
            y, list
        ), f'The transform BinariesToCategory only accepts label list as input, but obtain {type(y)} as input.'
        return sum([v * 2 ** i for i, v in enumerate(reversed(y))])


class BaselineRemoval(EEGTransform):
    r'''
    A transform method to subtract the baseline signal (the signal recorded before the emotional stimulus), the nosie signal is removed from the emotional signal unrelated to the emotional stimulus.

    TorchEEG recommends using this class in online_transform for higher processing speed. Even though, this class is also supported in offline_transform. Usually, the baseline needs the same transformation as the experimental signal, please add :obj:`apply_to_baseline=True` to all transforms before this operation to ensure that the transformation is performed on the baseline signal

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Compose([
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.ToTensor(apply_to_baseline=True),
            transforms.BaselineRemoval(),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ])

        t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg'].shape
        >>> (4, 9, 9)

    .. automethod:: __call__
    '''

    def __init__(self):
        # super(BaselineRemoval, self).__init__(apply_to_baseline=False)
        super().__init__(apply_to_baseline=False)

    def __call__(self, *args, eeg: any, baseline: Union[any, None] = None, **kwargs) -> Dict[str, any]:
        r'''
        Args:
            eeg (any): The input EEG signal.
            baseline (any) : The corresponding baseline signal.

        Returns:
            any: The transformed result after removing the baseline signal.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: any, **kwargs) -> any:
        if kwargs['baseline'] is None:
            return eeg

        assert kwargs[
                   'baseline'].shape == eeg.shape, f'The shape of baseline signals ({kwargs["baseline"].shape}) need to be consistent with the input signal ({eeg.shape}). Did you forget to add apply_to_baseline=True to the transforms before BaselineRemoval so that these transforms are applied to the baseline signal simultaneously?'
        return eeg - kwargs['baseline']

    @property
    def targets_as_params(self) -> List[str]:
        return ['baseline']

    def get_params_dependent_on_targets(self, params: Dict[str, any]) -> Dict[str, any]:
        return {'baseline': params['baseline']}



class Lambda(BaseTransform):
    r'''
    Apply a user-defined lambda as a transform.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Lambda(targets=['y'], lambda x: x + 1)
        t(y=1)['y']
        >>> 2

    Args:
        targets (list): What data to transform via the Lambda. (default: :obj:`['eeg', 'baseline', 'y']`)
        lambd (Callable): Lambda/function to be used for transform.

    .. automethod:: __call__
    '''
    def __init__(self,
                 lambd: Callable,
                 targets: List[str] = ['eeg', 'baseline', 'y']):
        super(Lambda, self).__init__()
        self._targets = targets
        self.lambd = lambd

    @property
    def targets(self) -> Dict[str, Callable]:
        return {target: self.apply for target in self._targets}

    def apply(self, *args, **kwargs) -> any:
        r'''
        Args:
            x (any): The input.

        Returns:
            any: The transformed output.
        '''
        return self.lambd(args[0])

    def __call__(self, *args, **kwargs) -> Dict[str, any]:
        r'''
        Args:
            x (any): The input.
        Returns:
            any: The transformed output.
        '''
        return super().__call__(*args, **kwargs)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'lambd': self.lambd,
            'targets': [...]
        })


class MeanStdNormalize(EEGTransform):
    r'''
    Perform z-score normalization on the input data. This class allows the user to define the dimension of normalization and the used statistic.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.MeanStdNormalize(axis=0)
        # normalize along the first dimension (electrode dimension)
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        t = transforms.MeanStdNormalize(axis=1)
        # normalize along the second dimension (temproal dimension)
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        mean (np.array, optional): The mean used in the normalization process, allowing the user to provide mean statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        std (np.array, optional): The standard deviation used in the normalization process, allowing the user to provide tandard deviation statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __init__(self,
                 mean: Union[np.ndarray, None] = None,
                 std: Union[np.ndarray, None] = None,
                 axis: Union[int, None] = None,
                 apply_to_baseline: bool = False):
        super(MeanStdNormalize,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.mean = mean
        self.std = std
        self.axis = axis

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals or features.
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The normalized results.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs):
        if (self.mean is None) or (self.std is None):
            if self.axis is None:
                mean = eeg.mean()
                std = eeg.std()
            else:
                mean = eeg.mean(axis=self.axis, keepdims=True)
                std = eeg.std(axis=self.axis, keepdims=True)
        else:
            if self.axis is None:
                axis = 1
            else:
                axis = self.axis
            assert len(self.mean) == eeg.shape[
                axis], f'The given normalized axis has {eeg.shape[axis]} dimensions, which does not match the given mean\'s dimension {len(self.mean)}.'
            assert len(self.std) == eeg.shape[
                axis], f'The given normalized axis has {eeg.shape[axis]} dimensions, which does not match the given std\'s dimension {len(self.std)}.'
            shape = [1] * len(eeg.shape)
            shape[axis] = -1
            mean = self.mean.reshape(*shape)
            std = self.std.reshape(*shape)
        return (eeg - mean) / std

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'mean': self.mean,
            'std': self.std,
            'axis': self.axis
        })


class MinMaxNormalize(EEGTransform):
    r'''
    Perform min-max normalization on the input data. This class allows the user to define the dimension of normalization and the used statistic.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.MinMaxNormalize(axis=0)
        # normalize along the first dimension (electrode dimension)
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        from torcheeg import transforms

        t = transforms.MinMaxNormalize(axis=1)
        # normalize along the second dimension (temproal dimension)
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        min (np.array, optional): The minimum used in the normalization process, allowing the user to provide minimum statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        max (np.array, optional): The maximum used in the normalization process, allowing the user to provide maximum statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __init__(self,
                 min: Union[np.ndarray, None, float] = None,
                 max: Union[np.ndarray, None, float] = None,
                 axis: Union[int, None] = None,
                 apply_to_baseline: bool = False):
        super(MinMaxNormalize,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.min = min
        self.max = max
        self.axis = axis

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals or features.
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The normalized results.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        if (self.min is None) or (self.max is None):
            # if not given min/max
            if self.axis is None:
                # calc overall min/max
                min = eeg.min()
                max = eeg.max()
            else:
                # calc axis min/max
                min = eeg.min(axis=self.axis, keepdims=True)
                max = eeg.max(axis=self.axis, keepdims=True)
        else:
            if self.axis is None:
                # given overall min/max
                assert isinstance(self.min, float) and isinstance(
                    self.max, float
                ), f'The given normalized axis is None, which requires a float number as min/max to normalize the samples, but get {type(self.min)} and {type(self.max)}.'

                min = self.min
                max = self.max
            else:
                # given axis min/max
                axis = self.axis

                assert len(self.min) == eeg.shape[
                    axis], f'The given normalized axis has {eeg.shape[axis]} dimensions, which does not match the given min\'s dimension {len(self.min)}.'
                assert len(self.max) == eeg.shape[
                    axis], f'The given normalized axis has {eeg.shape[axis]} dimensions, which does not match the given max\'s dimension {len(self.max)}.'

                shape = [1] * len(eeg.shape)
                shape[axis] = -1
                min = self.min.reshape(*shape)
                max = self.max.reshape(*shape)

        return (eeg - min) / (max - min)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'min': self.min,
            'max': self.max,
            'axis': self.axis
        })


# import numpy as np
# import torch
#
# from typing import List, Union


def after_hook_normalize(
        data: List[Union[np.ndarray, torch.Tensor]],
        eps: float = 1e-6) -> List[Union[np.ndarray, torch.Tensor]]:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject after dividing it into chunks and transforming the divided chunks.

    It is used as follows:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_normalize

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=after_hook_normalize,
                              num_worker=4,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_normalize

        DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=partial(after_hook_normalize, eps=1e-5),
                              num_worker=4,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    Args:
        data (list): A list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
        eps (float): The term added to the denominator to improve numerical stability (default: :obj:`1e-6`)

    Returns:
        list: The normalized results of a trial. It is a list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
    '''
    if isinstance(data[0], np.ndarray):
        data = np.stack(data, axis=0)

        min_v = data.min(axis=0, keepdims=True)
        max_v = data.max(axis=0, keepdims=True)
        data = (data - min_v) / (max_v - min_v + eps)

        return [sample for sample in data]
    elif isinstance(data[0], torch.Tensor):
        data = torch.stack(data, dim=0)

        min_v, _ = data.min(axis=0, keepdims=True)
        max_v, _ = data.max(axis=0, keepdims=True)
        data = (data - min_v) / (max_v - min_v + eps)

        return [sample for sample in data]
    else:
        raise ValueError(
            'The after_hook_normalize only supports np.ndarray and torch.Tensor. Please make sure the outputs of offline_transform ({}) are np.ndarray or torch.Tensor.'
            .format(type(data[0])))


def after_hook_running_norm(
        data: List[Union[np.ndarray, torch.Tensor]],
        decay_rate: float = 0.9,
        eps: float = 1e-6) -> List[Union[np.ndarray, torch.Tensor]]:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject after dividing it into chunks and transforming the divided chunks.

    It is used as follows:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_running_norm
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=after_hook_running_norm,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_running_norm

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=partial(after_hook_running_norm, decay_rate=0.9, eps=1e-6),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    Args:
        data (list): A list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
        decay_rate (float): The decay rate used in the running normalization (default: :obj:`0.9`)
        eps (float): The term added to the denominator to improve numerical stability (default: :obj:`1e-6`)

    Returns:
        list: The normalized results of a trial. It is a list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
    '''
    if isinstance(data[0], np.ndarray):
        data = np.stack(data, axis=0)

        running_mean = np.zeros_like(data[0])
        running_var = np.zeros_like(data[0])

        for i, current_sample in enumerate(data):
            running_mean = decay_rate * running_mean + (
                    1 - decay_rate) * current_sample
            running_var = decay_rate * running_var + (
                    1 - decay_rate) * np.square(current_sample - running_mean)
            data[i] = (data[i] - running_mean) / np.sqrt(running_var + eps)

        return [sample for sample in data]
    elif isinstance(data[0], torch.Tensor):
        data = torch.stack(data, dim=0)

        running_mean = torch.zeros_like(data[0])
        running_var = torch.zeros_like(data[0])

        for i, current_sample in enumerate(data):
            running_mean = decay_rate * running_mean + (
                    1 - decay_rate) * current_sample
            running_var = decay_rate * running_var + (
                    1 - decay_rate) * torch.square(current_sample - running_mean)
            data[i] = (data[i] - running_mean) / torch.sqrt(running_var + eps)

        return [sample for sample in data]
    else:
        raise ValueError(
            'The after_hook_running_norm only supports np.ndarray and torch.Tensor. Please make sure the outputs of offline_transform ({}) are np.ndarray or torch.Tensor.'
            .format(type(data[0])))


def after_hook_linear_dynamical_system(
        data: List[Union[np.ndarray, torch.Tensor]],
        V0: float = 0.01,
        A: float = 1,
        T: float = 0.0001,
        C: float = 1,
        sigma: float = 1) -> List[Union[np.ndarray, torch.Tensor]]:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject after dividing it into chunks and transforming the divided chunks.

    It is used as follows:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_linear_dynamical_system

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=after_hook_linear_dynamical_system,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_linear_dynamical_system

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=partial(after_hook_linear_dynamical_system, V0=0.01, A=1, T=0.0001, C=1, sigma=1),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    Args:
        data (list): A list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
        V0 (float): The initial variance of the linear dynamical system (default: :obj:`0.01`)
        A (float): The coefficient of the linear dynamical system (default: :obj:`1`)
        T (float): The term added to the diagonal of the covariance matrix (default: :obj:`0.0001`)
        C (float): The coefficient of the linear dynamical system (default: :obj:`1`)
        sigma (float): The variance of the linear dynamical system (default: :obj:`1`)

    Returns:
        list: The normalized results of a trial. It is a list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
    '''
    if isinstance(data[0], np.ndarray):
        # save the data[0].shape and flatten them
        shape = data[0].shape
        data = np.stack([sample.flatten() for sample in data], axis=0)

        ave = np.mean(data, axis=0)
        u0 = ave
        X = data.transpose((1, 0))

        [m, n] = X.shape
        P = np.zeros((m, n))
        u = np.zeros((m, n))
        V = np.zeros((m, n))
        K = np.zeros((m, n))

        K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m,))
        u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
        V[:, 0] = (np.ones((m,)) - K[:, 0] * C) * V0

        for i in range(1, n):
            P[:, i - 1] = A * V[:, i - 1] * A + T
            K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
            u[:,
            i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
            V[:, i] = (np.ones((m,)) - K[:, i] * C) * P[:, i - 1]

        X = u

        return [sample.reshape(shape) for sample in X.transpose((1, 0))]

    elif isinstance(data[0], torch.Tensor):
        shape = data[0].shape
        data = torch.stack([sample.flatten() for sample in data], dim=0)

        ave = torch.mean(data, dim=0)
        u0 = ave
        X = data.transpose(1, 0)

        [m, n] = X.shape
        P = torch.zeros((m, n))
        u = torch.zeros((m, n))
        V = torch.zeros((m, n))
        K = torch.zeros((m, n))

        K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * torch.ones((m,))
        u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
        V[:, 0] = (torch.ones((m,)) - K[:, 0] * C) * V0

        for i in range(1, n):
            P[:, i - 1] = A * V[:, i - 1] * A + T
            K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
            u[:,
            i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
            V[:, i] = (torch.ones((m,)) - K[:, i] * C) * P[:, i - 1]

        X = u

        return [sample.reshape(shape) for sample in X.transpose(1, 0)]

    else:
        raise ValueError(
            'The after_hook_linear_dynamical_system only supports np.ndarray and torch.Tensor. Please make sure the outputs of offline_transform ({}) are np.ndarray or torch.Tensor.'
            .format(type(data[0])))


class PickElectrode(EEGTransform):
    r'''
    Select parts of electrode signals based on a given electrode index list.

    .. code-block:: python

        from torcheeg import transforms
        from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

        t = transforms.PickElectrode(transforms.PickElectrode.to_index_list(
            ['FP1', 'AF3', 'F3', 'F7',
             'FC5', 'FC1', 'C3', 'T7',
             'CP5', 'CP1', 'P3', 'P7',
             'PO3','O1', 'FP2', 'AF4',
             'F4', 'F8', 'FC6', 'FC2',
             'C4', 'T8', 'CP6', 'CP2',
             'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST))
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (28, 128)

    Args:
        pick_list (np.ndarray): Selected electrode list. Should consist of integers representing the corresponding electrode indices. :obj:`to_index_list` can be used to obtain an index list when we only know the names of the electrode and not their indices.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self, pick_list: List[int], apply_to_baseline: bool = False):
        super(PickElectrode, self).__init__(apply_to_baseline=apply_to_baseline)
        self.pick_list = pick_list

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The output signals with the shape of [number of picked electrodes, number of data points].
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        assert max(self.pick_list) < eeg.shape[
            0], f'The index {max(self.pick_list)} of the specified electrode is out of bounds {eeg.shape[0]}.'
        return eeg[self.pick_list]

    @staticmethod
    def to_index_list(electrode_list: List[str],
                      dataset_electrode_list: List[str],
                      strict_mode=False) -> List[int]:
        r'''
        Args:
            electrode_list (list): picked electrode name, consisting of strings.
            dataset_electrode_list (list): The description of the electrode information contained in the EEG signal in the dataset, consisting of strings. For the electrode position information, please refer to constants grouped by dataset :obj:`datasets.constants`.
            strict_mode: (bool): Whether to use strict mode. In strict mode, unmatched picked electrode names are thrown as errors. Otherwise, unmatched picked electrode names are automatically ignored. (default: :obj:`False`)
        Returns:
            list: Selected electrode list, consisting of integers representing the corresponding electrode indices.
        '''
        dataset_electrode_dict = dict(
            zip(dataset_electrode_list,
                list(range(len(dataset_electrode_list)))))
        if strict_mode:
            return [
                dataset_electrode_dict[electrode]
                for electrode in electrode_list
            ]
        return [
            dataset_electrode_dict[electrode] for electrode in electrode_list
            if electrode in dataset_electrode_dict
        ]

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'pick_list': [...]
        })
from typing import Dict, Sequence, Union

import torch
from torch.nn.functional import interpolate
class Resize(EEGTransform):
    r'''
    Use an interpolation algorithm to scale a grid-like EEG signal at the spatial dimension.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.ToTensor(size=(64, 64))
        t(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 64, 64)

    Args:
        size (tuple): The output spatial size.
        interpolation (str): The interpolation algorithm used for upsampling, can be nearest, linear, bilinear, bicubic, trilinear, and area. (default: :obj:`'nearest'`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 size: Union[Sequence[int], int],
                 interpolation: str = "bilinear",
                 apply_to_baseline: bool = False):
        super(Resize, self).__init__(apply_to_baseline=apply_to_baseline)
        self.size = size
        self.interpolation = interpolation

    def __call__(self,
                 *args,
                 eeg: torch.Tensor,
                 baseline: Union[torch.Tensor, None] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        r'''
        Args:
            eeg (torch.Tensor): The input EEG signal in shape of [height of grid, width of grid, number of data points].
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            torch.Tensor[new height of grid, new width of grid, number of sub-bands]: The scaled EEG signal at the saptial dimension.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        assert eeg.dim() == 3, f'The Resize only allows to input a 3-d tensor, but the input has dimension {eeg.dim()}'

        eeg = eeg.unsqueeze(0)

        align_corners = False if self.interpolation in ["bilinear", "bicubic"] else None

        interpolated_x = interpolate(eeg, size=self.size, mode=self.interpolation, align_corners=align_corners)

        return interpolated_x.squeeze(0)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'size': self.size, 'interpolation': self.interpolation})