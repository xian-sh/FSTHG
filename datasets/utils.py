import atexit
import logging
import os
import random
import shutil
import ssl
import string
import sys
import tempfile
import time
import urllib
from functools import partial
from typing import Optional
from typing import Any, Callable, Dict, Union
import torch
import pandas as pd
import pickle
import lmdb
import csv

log = logging.getLogger('torcheeg')


class _EEGSignalIO:

    @property
    def write_pointer(self):
        return len(self)

    def keys(self):
        raise NotImplementedError

    def eegs(self):
        raise NotImplementedError

    def read_eeg(self, key: str) -> any:
        raise NotImplementedError

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        raise NotImplementedError


class MemoryEEGSignalIO(_EEGSignalIO):

    def __init__(self):
        self._memory = {}

    def __len__(self):
        return len(self._memory)

    def keys(self):
        r'''
        Get all keys in the MemoryEEGSignalIO.

        Returns:
            list: The list of keys in the MemoryEEGSignalIO.
        '''
        return list(self._memory.keys())

    def eegs(self):
        return list(self._memory.values())

    def read_eeg(self, key: str) -> any:
        r'''
        Read all the MemoryEEGSignalIO into memory, and index the specified EEG signal in memory with the given :obj:`key`.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        if key not in self._memory:
            raise RuntimeError(
                f'Unable to index the EEG signal sample with key {key}!')

        return self._memory[key]

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        r'''
        Write EEG signal to memory.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str): The key of the EEG signal to be inserted, if not specified, it will be an auto-incrementing
        '''
        if key is None:
            key = str(self.write_pointer)

        if eeg is None:
            raise RuntimeError(f'Save None to the memory with the key {key}!')

        self._memory[key] = eeg

        return key

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class LMDBEEGSignalIO(_EEGSignalIO):

    def __init__(self, io_path: str, io_size: int = 1048576) -> None:
        self.io_path = io_path
        self.io_size = io_size

        if not os.path.exists(self.io_path):
            os.makedirs(self.io_path, exist_ok=True)
        self._env = lmdb.open(path=self.io_path,
                              map_size=self.io_size,
                              lock=False)

    def __del__(self):
        self._env.close()

    def __len__(self):
        with self._env.begin(write=False) as transaction:
            return transaction.stat()['entries']

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        r'''
        Write EEG signal to database.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str, optional): The key of the EEG signal to be inserted, if not specified, it will be an auto-incrementing integer.

        Returns:
            int: The index of written EEG signals in the database.
        '''

        if key is None:
            key = str(self.write_pointer)

        if eeg is None:
            raise RuntimeError(f'Save None to the LMDB with the key {key}!')

        try_again = False
        try:
            with self._env.begin(write=True) as transaction:
                transaction.put(key.encode(), pickle.dumps(eeg))
        except lmdb.MapFullError:
            self.io_size = self.io_size * 2
            self._env.set_mapsize(self.io_size)
            try_again = True
        if try_again:
            return self.write_eeg(key=key, eeg=eeg)
        return key

    def read_eeg(self, key: str) -> any:
        r'''
        Query the corresponding EEG signal in the database according to the index.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        with self._env.begin(write=False) as transaction:
            eeg = transaction.get(key.encode())

        if eeg is None:
            raise RuntimeError(
                f'Unable to index the EEG signal sample with key {key}!')

        return pickle.loads(eeg)

    def keys(self):
        r'''
        Get all keys in the LMDBEEGSignalIO.

        Returns:
            list: The list of keys in the LMDBEEGSignalIO.
        '''
        with self._env.begin(write=False) as transaction:
            return [
                key.decode()
                for key in transaction.cursor().iternext(keys=True,
                                                         values=False)
            ]

    def eegs(self):
        r'''
        Get all EEG signals in the LMDBEEGSignalIO.

        Returns:
            list: The list of EEG signals in the LMDBEEGSignalIO.
        '''
        return [self.read_eeg(key) for key in self.keys()]

    def __getstate__(self):
        # pickle for Pallarel
        state = self.__dict__.copy()
        del state['_env']
        return state

    def __setstate__(self, state):
        # pickle for Pallarel
        self.__dict__.update(state)
        self._env = lmdb.open(path=self.io_path,
                              map_size=self.io_size,
                              lock=False)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items() if k != '_env'
        })
        result._env = lmdb.open(path=self.io_path,
                                map_size=self.io_size,
                                lock=False)
        return result


class PickleEEGSignalIO(_EEGSignalIO):

    def __init__(self, io_path: str) -> None:
        self.io_path = io_path

        os.makedirs(self.io_path, exist_ok=True)

    def __len__(self):
        return len(os.listdir(self.io_path))

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        r'''
            Write EEG signal to folder.

            Args:
                eeg (any): EEG signal samples to be written into the folder.
                key (str, optional): The key of the EEG signal to be inserted, if not specified, it will be an auto-incrementing integer.

            Returns:
                int: The index of written EEG signals in the folder.
            '''

        if key is None:
            key = str(self.write_pointer)

        if eeg is None:
            raise RuntimeError(f'Save None to the LMDB with the key {key}!')

        with open(os.path.join(self.io_path, key), 'wb') as f:
            pickle.dump(eeg, f)

        return key

    def read_eeg(self, key: str) -> any:
        r'''
            Query the corresponding EEG signal in the folder according to the index.

            Args:
                key (str): The index of the EEG signal to be queried.

            Returns:
                any: The EEG signal sample.
            '''
        with open(os.path.join(self.io_path, key), 'rb') as f:
            eeg = pickle.load(f)

        return eeg

    def keys(self):
        r'''
            Get all keys in the PickleEEGSignalIO.

            Returns:
                list: The list of keys in the PickleEEGSignalIO.
            '''
        return os.listdir(self.io_path)

    def eegs(self):
        r'''
            Get all EEG signals in the PickleEEGSignalIO.

            Returns:
                list: The list of EEG signals in the PickleEEGSignalIO.
            '''
        return [self.read_eeg(key) for key in self.keys()]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class EEGSignalIO:
    r'''
    A general-purpose, lightweight and efficient EEG signal IO APIs for converting various real-world EEG signal datasets into samples and storing them in the database. Here, we draw on the implementation ideas of industrial-grade application Caffe, and encapsulate a set of EEG signal reading and writing methods based on Lightning Memory-Mapped Database (LMDB), which not only unifies the differences of data types in different databases, but also accelerates the reading of data during training and testing.

    .. code-block:: python

        eeg_io = EEGSignalIO('YOUR_PATH')
        key = eeg_io.write_eeg(np.random.randn(32, 128))
        eeg = eeg_io.read_eeg(key)
        eeg.shape
        >>> (32, 128)

    Args:
        io_path (str): Where the database is stored.
        io_size (int, optional): The maximum capacity of the database. It will increase according to the size of the dataset. (default: :obj:`1024`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems. Here, a file system based and a memory based EEG signal storages are also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
    '''

    def __init__(self,
                 io_path: str,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb') -> None:
        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode

        if self.io_mode == 'lmdb':
            self._io = LMDBEEGSignalIO(io_path=self.io_path,
                                       io_size=self.io_size)
        elif self.io_mode == 'pickle':
            self._io = PickleEEGSignalIO(io_path=self.io_path)
        elif self.io_mode == 'memory':
            self._io = MemoryEEGSignalIO()
        else:
            raise RuntimeError(
                f'Unsupported io_mode {self.io_mode}, please choose from lmdb, pickle and memory.'
            )

    def __del__(self):
        del self._io

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items() if k != '_io'
        })
        result._io = self._io.__copy__()
        return result

    def __len__(self):
        return len(self._io)

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        r'''
        Write EEG signal to database.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str, optional): The key of the EEG signal to be inserted, if not specified, it will be an auto-incrementing integer.

        Returns:
            int: The index of written EEG signals in the database.
        '''

        return self._io.write_eeg(eeg=eeg, key=key)

    def read_eeg(self, key: str) -> any:
        r'''
        Query the corresponding EEG signal in the database according to the index.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        return self._io.read_eeg(key)

    def keys(self):
        r'''
        Get all keys in the EEGSignalIO.

        Returns:
            list: The list of keys in the EEGSignalIO.
        '''
        return self._io.keys()

    def eegs(self):
        r'''
        Get all EEG signals in the EEGSignalIO.

        Returns:
            list: The list of EEG signals in the EEGSignalIO.
        '''
        return self._io.eegs()

    def to_lmdb(self, io_path: str, io_size: int = 1048576):
        r'''
        Convert to the LMDBEEGSignalIO, where the index of each sample in the database corresponds to the key, and the EEG signal stored in the database corresponds to the value.
        '''
        _io = LMDBEEGSignalIO(io_path=io_path, io_size=io_size)

        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = 'lmdb'

        for key in self.keys():
            _io.write_eeg(self.read_eeg(key=key), key=key)

        self._io = _io

    def to_pickle(self, io_path: str):
        r'''
        Convert to the PickleEEGSignalIO, where the index of each sample in the database corresponds to the key, and the EEG signal stored in the database corresponds to the value.
        '''
        _io = PickleEEGSignalIO(io_path=io_path)

        self.io_path = io_path
        self.io_mode = 'pickle'

        for key in self.keys():
            _io.write_eeg(self.read_eeg(key=key), key=key)

        self._io = _io

    def to_memory(self):
        r'''
        Convert to the MemoryEEGSignalIO, where the index of each sample in the database corresponds to the key, and the EEG signal stored in the database corresponds to the value.
        '''
        _io = MemoryEEGSignalIO()

        self.io_mode = 'memory'

        for key in self.keys():
            _io.write_eeg(self.read_eeg(key=key), key=key)

        self._io = _io



class MetaInfoIO:
    r'''
    Use with torcheeg.io.EEGSignalIO to store description information for EEG signals in the form of a table, so that the user can still analyze, insert, delete and modify the corresponding information after the generation is completed.

    .. code-block:: python

        info_io = MetaInfoIO('YOUR_PATH')
        key = info_io.write_info({
            'clip_id': 0,
            'baseline_id': 1,
            'valence': 1.0,
            'arousal': 9.0
        })
        info = info_io.read_info(key).to_dict()
        >>> {
                'clip_id': 0,
                'baseline_id': 1,
                'valence': 1.0,
                'arousal': 9.0
            }

    Args:
        io_path (str): Where the table is stored.
    '''

    def __init__(self, io_path: str) -> None:
        self.io_path = io_path
        if not os.path.exists(self.io_path):
            os.makedirs(os.path.dirname(io_path), exist_ok=True)
            open(self.io_path, 'x').close()
            self.write_pointer = 0
        else:
            self.write_pointer = len(self)

    def __len__(self):
        if os.path.getsize(self.io_path) == 0:
            return 0
        info_list = pd.read_csv(self.io_path)
        return len(info_list)

    def write_info(self, obj: Dict) -> int:
        r'''
        Insert a description of the EEG signal.

        Args:
            obj (dict): The description to be written into the table.

        Returns:
            int: The index of written EEG description in the table.
        '''
        with open(self.io_path, 'a+') as f:
            require_head = os.path.getsize(self.io_path) == 0
            writer = csv.DictWriter(f, fieldnames=list(obj.keys()))
            if require_head:
                writer.writeheader()
            writer.writerow(obj)
        key = self.write_pointer
        self.write_pointer += 1
        return key

    def read_info(self, key) -> pd.DataFrame:
        r'''
        Query the corresponding EEG description in the table according to the index.

        Args:
            key (int): The index of the EEG description to be queried.
        Returns:
            pd.DataFrame: The EEG description.
        '''
        return pd.read_csv(self.io_path).iloc[key]

    def read_all(self) -> pd.DataFrame:
        r'''
        Get all EEG descriptions in the database in tabular form.

        Returns:
            pd.DataFrame: The EEG descriptions.
        '''
        if os.path.getsize(self.io_path) == 0:
            return pd.DataFrame()
        return pd.read_csv(self.io_path)


def get_temp_dir_path():
    temp_dir_path = tempfile.mkdtemp()
    atexit.register(partial(shutil.rmtree, temp_dir_path, ignore_errors=True))
    return temp_dir_path


def get_random_dir_path(root_path='.torcheeg', dir_prefix='tmp'):
    """
    Generates a unique folder name based on the current timestamp and a random suffix.
    The folder name is intended to be used for creating a temporary cache folder in torcheeg.

    Returns:
        str: A unique folder name.
    """
    timestamp = int(time.time() * 1000)

    random_suffix = ''.join(
        random.choices(string.ascii_letters + string.digits, k=5))

    dir_path = f"{dir_prefix}_{timestamp}_{random_suffix}"

    return os.path.join(root_path, dir_path)


def get_package_dir_path():
    home_dir = None

    if 'nt' == os.name.lower():
        if os.path.isdir(os.path.join(os.getenv('APPDATA'), '.torcheeg')):
            home_dir = os.getenv('APPDATA')
        else:
            home_dir = os.getenv('USERPROFILE')
    else:
        home_dir = os.path.expanduser('~')

    if home_dir is None:
        raise RuntimeError(
            'Cannot resolve home dictionary, please report this bug to TorchEEG Teams.'
        )

    return os.path.join(home_dir, '.torcheeg')


def download_url(url: str,
                 folder: str,
                 verbose: bool = True,
                 filename: Optional[str] = None):
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = os.path.join(folder, filename)

    if os.path.exists(path):  # pragma: no cover
        if verbose:
            log.info(f'Using existing file {filename}', file=sys.stderr)
        return path

    if verbose:
        log.info(f'Downloading {url}', file=sys.stderr)

    os.makedirs(os.path.expanduser(os.path.normpath(folder)))

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path