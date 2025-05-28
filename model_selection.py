import logging
import os
from copy import copy
import re

from typing import Dict, Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn import model_selection

from datasets.base import BaseDataset

from datasets.utils import get_random_dir_path

log = logging.getLogger('torcheeg')


def train_test_split(dataset: BaseDataset,
                     test_size: float = 0.2,
                     shuffle: bool = False,
                     random_state: Union[float, None] = None,
                     split_path: Union[None, str] = None):
    r'''
    A tool function for cross-validations, to divide the training set and the test set. It is suitable for experiments with large dataset volume and no need to use k-fold cross-validations. The test samples are sampled according to a certain proportion, and other samples are used as training samples. In most literatures, 20% of the data are sampled for testing.

    :obj:`train_test_split` devides the training set and the test set without grouping. It means that during random sampling, adjacent signal samples may be assigned to the training set and the test set, respectively. When random sampling is not used, some subjects are not included in the training set. If you think these situations shouldn't happen, consider using :obj:`train_test_split_per_subject_groupby_trial` or :obj:`train_test_split_groupby_trial`.

    .. image:: _static/train_test_split.png
        :alt: The schematic diagram of train_test_split
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        train_dataset, test_dataset = train_test_split(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (int):  If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. (default: :obj:`0.2`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    if not os.path.exists(split_path):
        log.info(f'📊 | Create the split of train and test set.')
        log.info(
            f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
        )
        os.makedirs(split_path)
        info = dataset.info

        n_samples = len(dataset)
        indices = np.arange(n_samples)
        train_index, test_index = model_selection.train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle)
        train_info = info.iloc[train_index]
        test_info = info.iloc[test_index]

        train_info.to_csv(os.path.join(split_path, 'train.csv'), index=False)
        test_info.to_csv(os.path.join(split_path, 'test.csv'), index=False)

    else:
        log.info(
            f'📊 | Detected existing split of train and test set, use existing split from {split_path}.'
        )
        log.info(
            f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
        )

    train_info = pd.read_csv(os.path.join(split_path, 'train.csv'))
    test_info = pd.read_csv(os.path.join(split_path, 'test.csv'))

    train_dataset = copy(dataset)
    train_dataset.info = train_info

    test_dataset = copy(dataset)
    test_dataset.info = test_info

    return train_dataset, test_dataset


def train_test_split_cross_trial(dataset: BaseDataset,
                                 test_size: float = 0.2,
                                 shuffle: bool = False,
                                 random_state: Union[float, None] = None,
                                 split_path: Union[None, str] = None):
    r'''
    A tool function for cross-validations, to divide the training set and the test set. It is suitable for experiments with large dataset volume and no need to use k-fold cross-validations. Parts of trials are sampled according to a certain proportion as the test dataset, and samples from other trials are used as training samples. In most literatures, 20% of the data are sampled for testing.

    :obj:`train_test_split_cross_trial` devides training set and the test set at the dimension of each trial. For example, when :obj:`test_size=0.2`, the first 80% of samples of each trial are used for training, and the last 20% of samples are used for testing. It is more consistent with real applications and can test the generalization of the model to a certain extent.

    .. image:: _static/train_test_split_cross_trial.png
        :alt: The schematic diagram of train_test_split_cross_trial
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_cross_trial
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        train_dataset, test_dataset = train_test_split_cross_trial(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (int):  If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. (default: :obj:`0.2`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    if not os.path.exists(split_path):
        log.info(f'📊 | Create the split of train and test set.')
        log.info(
            f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
        )
        os.makedirs(split_path)
        info = dataset.info
        subjects = list(set(info['subject_id']))

        train_info = None
        test_info = None

        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            trial_ids = list(set(subject_info['trial_id']))

            train_index_trial_ids, test_index_trial_ids = model_selection.train_test_split(
                trial_ids,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state)

            if len(train_index_trial_ids) == 0 or len(
                    test_index_trial_ids) == 0:
                raise ValueError(
                    f'The number of training or testing trials for subject {subject} is zero.'
                )

            train_trial_ids = np.array(
                trial_ids)[train_index_trial_ids].tolist()
            test_trial_ids = np.array(trial_ids)[test_index_trial_ids].tolist()

            subject_train_info = []
            for train_trial_id in train_trial_ids:
                subject_train_info.append(
                    subject_info[subject_info['trial_id'] == train_trial_id])
            subject_train_info = pd.concat(subject_train_info,
                                           ignore_index=True)

            subject_test_info = []
            for test_trial_id in test_trial_ids:
                subject_test_info.append(
                    subject_info[subject_info['trial_id'] == test_trial_id])
            subject_test_info = pd.concat(subject_test_info, ignore_index=True)

            if train_info is None and test_info is None:
                train_info = [subject_train_info]
                test_info = [subject_test_info]
            else:
                train_info.append(subject_train_info)
                test_info.append(subject_test_info)

        train_info = pd.concat(train_info, ignore_index=True)
        test_info = pd.concat(test_info, ignore_index=True)

        train_info.to_csv(os.path.join(split_path, 'train.csv'), index=False)
        test_info.to_csv(os.path.join(split_path, 'test.csv'), index=False)

    else:
        log.info(
            f'📊 | Detected existing split of train and test set, use existing split from {split_path}.'
        )
        log.info(
            f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
        )

    train_info = pd.read_csv(os.path.join(split_path, 'train.csv'))
    test_info = pd.read_csv(os.path.join(split_path, 'test.csv'))

    train_dataset = copy(dataset)
    train_dataset.info = train_info

    test_dataset = copy(dataset)
    test_dataset.info = test_info

    return train_dataset, test_dataset



def train_test_split_groupby_trial(dataset: BaseDataset,
                                   test_size: float = 0.2,
                                   shuffle: bool = False,
                                   random_state: Union[float, None] = None,
                                   split_path: Union[None, str] = None):
    r'''
    A tool function for cross-validations, to divide the training set and the test set. It is suitable for experiments with large dataset volume and no need to use k-fold cross-validations. The test samples are sampled according to a certain proportion, and other samples are used as training samples. In most literatures, 20% of the data are sampled for testing.

    :obj:`train_test_split_groupby_trial` devides training set and the test set at the dimension of each trial. For example, when :obj:`test_size=0.2`, the first 80% of samples of each trial are used for training, and the last 20% of samples are used for testing. It is more consistent with real applications and can test the generalization of the model to a certain extent.

    .. image:: _static/train_test_split_groupby_trial.png
        :alt: The schematic diagram of train_test_split_groupby_trial
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_groupby_trial
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        train_dataset, test_dataset = train_test_split_groupby_trial(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (int):  If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. (default: :obj:`0.2`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    if not os.path.exists(split_path):
        log.info(f'📊 | Create the split of train and test set.')
        log.info(
            f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
        )
        os.makedirs(split_path)
        info = dataset.info
        subjects = list(set(info['subject_id']))
        trial_ids = list(set(info['trial_id']))

        train_info = None
        test_info = None

        for subject in subjects:
            for trial_id in trial_ids:
                cur_info = info[(info['subject_id'] == subject)
                                & (info['trial_id'] == trial_id)].reset_index()

                n_samples = len(cur_info)
                indices = np.arange(n_samples)
                train_index, test_index = model_selection.train_test_split(
                    indices,
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=shuffle)

                if train_info is None and test_info is None:
                    train_info = [cur_info.iloc[train_index]]
                    test_info = [cur_info.iloc[test_index]]
                else:
                    train_info.append(cur_info.iloc[train_index])
                    test_info.append(cur_info.iloc[test_index])

        train_info = pd.concat(train_info, ignore_index=True)
        test_info = pd.concat(test_info, ignore_index=True)

        train_info.to_csv(os.path.join(split_path, 'train.csv'), index=False)
        test_info.to_csv(os.path.join(split_path, 'test.csv'), index=False)

    else:
        log.info(
            f'📊 | Detected existing split of train and test set, use existing split from {split_path}.'
        )
        log.info(
            f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
        )

    train_info = pd.read_csv(os.path.join(split_path, 'train.csv'))
    test_info = pd.read_csv(os.path.join(split_path, 'test.csv'))

    train_dataset = copy(dataset)
    train_dataset.info = train_info

    test_dataset = copy(dataset)
    test_dataset.info = test_info

    return train_dataset, test_dataset


class KFoldPerSubjectCrossTrial:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject dependent experiments. Experiments were performed separately for each subject, where the data set is divided into k subsets of trials, with one subset trials being retained as the test set and the remaining k-1 subset trials being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    .. image:: _static/KFoldPerSubjectCrossTrial.png
        :alt: The schematic diagram of KFoldPerSubjectCrossTrial
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldPerSubjectCrossTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubjectCrossTrial(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset):
            # The total number of experiments is the number subjects multiplied by K
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    :obj:`KFoldPerSubjectCrossTrial` allows the user to specify the index of the subject of interest, when the user need to report the performance on each subject.

    .. code-block:: python

        from torcheeg.model_selection import KFoldPerSubjectCrossTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubjectCrossTrial(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset, subject=1):
            # k-fold cross-validation for subject 1
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[float, None] = None,
                 split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))

        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            trial_ids = list(set(subject_info['trial_id']))

            for fold_id, (train_index_trial_ids,
                          test_index_trial_ids) in enumerate(
                self.k_fold.split(trial_ids)):
                if len(train_index_trial_ids) == 0 or len(
                        test_index_trial_ids) == 0:
                    raise ValueError(
                        f'The number of training or testing trials for subject {subject} is zero.'
                    )

                train_trial_ids = np.array(
                    trial_ids)[train_index_trial_ids].tolist()
                test_trial_ids = np.array(
                    trial_ids)[test_index_trial_ids].tolist()

                train_info = []
                for train_trial_id in train_trial_ids:
                    train_info.append(subject_info[subject_info['trial_id'] ==
                                                   train_trial_id])
                train_info = pd.concat(train_info, ignore_index=True)

                test_info = []
                for test_trial_id in test_trial_ids:
                    test_info.append(
                        subject_info[subject_info['trial_id'] == test_trial_id])
                test_info = pd.concat(test_info, ignore_index=True)

                train_info.to_csv(os.path.join(
                    self.split_path,
                    f'train_subject_{subject}_fold_{fold_id}.csv'),
                    index=False)
                test_info.to_csv(os.path.join(
                    self.split_path,
                    f'test_subject_{subject}_fold_{fold_id}.csv'),
                    index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    @property
    def fold_ids(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(
                re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][1])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(
            self,
            dataset: BaseDataset,
            subject: Union[int,
            None] = None) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        subjects = self.subjects
        fold_ids = self.fold_ids

        if not subject is None:
            assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if (not subject is None) and (local_subject != subject):
                continue

            for fold_id in fold_ids:
                train_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'train_subject_{local_subject}_fold_{fold_id}.csv'))
                test_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'test_subject_{local_subject}_fold_{fold_id}.csv'))

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                test_dataset = copy(dataset)
                test_dataset.info = test_info

                yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

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

def train_test_split_per_subject_cross_trial(dataset: BaseDataset,
                                             test_size: float = 0.2,
                                             subject: str = 's01.dat',
                                             shuffle: bool = False,
                                             random_state: Union[float,
                                                                 None] = None,
                                             split_path: Union[None, str] = None):
    r'''
    A tool function for cross-validations, to divide the training set and the test set. It is suitable for subject dependent experiments with large dataset volume and no need to use k-fold cross-validations. For the first step, the EEG signal samples of the specified user are selected. Then, parts of trials are sampled according to a certain proportion as the test dataset, and samples from other trials are used as training samples. In most literatures, 20% of the data are sampled for testing.

    .. image:: _static/train_test_split_per_subject_cross_trial.png
        :alt: The schematic diagram of train_test_split_per_subject_cross_trial
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_per_subject_cross_trial
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        train_dataset, test_dataset = train_test_split_per_subject_cross_trial(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (int):  If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. (default: :obj:`0.2`)
        subject (str): The subject whose EEG samples will be used for training and test. (default: :obj:`s01.dat`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    if not os.path.exists(split_path):
        log.info(f'📊 | Create the split of train and test set.')
        log.info(
            f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
        )
        os.makedirs(split_path)
        info = dataset.info
        subjects = list(set(info['subject_id']))

        assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        subject_info = info[info['subject_id'] == subject]
        trial_ids = list(set(subject_info['trial_id']))

        train_trial_ids, test_trial_ids = model_selection.train_test_split(
            trial_ids,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state)

        if len(train_trial_ids) == 0 or len(test_trial_ids) == 0:
            raise ValueError(
                f'The number of training or testing trials for subject {subject} is zero.'
            )

        train_info = []
        for train_trial_id in train_trial_ids:
            train_info.append(
                subject_info[subject_info['trial_id'] == train_trial_id])
        train_info = pd.concat(train_info, ignore_index=True)

        test_info = []
        for test_trial_id in test_trial_ids:
            test_info.append(
                subject_info[subject_info['trial_id'] == test_trial_id])
        test_info = pd.concat(test_info, ignore_index=True)

        train_info.to_csv(os.path.join(split_path, 'train.csv'), index=False)
        test_info.to_csv(os.path.join(split_path, 'test.csv'), index=False)

    else:
        log.info(
            f'📊 | Detected existing split of train and test set, use existing split from {split_path}.'
        )
        log.info(
            f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
        )

    train_info = pd.read_csv(os.path.join(split_path, 'train.csv'))
    test_info = pd.read_csv(os.path.join(split_path, 'test.csv'))

    train_dataset = copy(dataset)
    train_dataset.info = train_info

    test_dataset = copy(dataset)
    test_dataset.info = test_info

    return train_dataset, test_dataset



def train_test_split_per_subject_groupby_trial(dataset: BaseDataset,
                                               test_size: float = 0.2,
                                               subject: str = 's01.dat',
                                               shuffle: bool = False,
                                               random_state: Union[float,
                                                                   None] = None,
                                               split_path: Union[None,
                                                                 str] = None):
    r'''
    A tool function for cross-validations, to divide the training set and the test set. It is suitable for subject dependent experiments with large dataset volume and no need to use k-fold cross-validations. For the first step, the EEG signal samples of the specified user are selected. Then, the test samples are sampled according to a certain proportion for each trial for this subject, and other samples are used as training samples. In most literatures, 20% of the data are sampled for testing.

    .. image:: _static/train_test_split_per_subject_groupby_trial.png
        :alt: The schematic diagram of train_test_split_per_subject_groupby_trial
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_per_subject_groupby_trial
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        train_dataset, test_dataset = train_test_split_per_subject_groupby_trial(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (int):  If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. (default: :obj:`0.2`)
        subject (str): The subject whose EEG samples will be used for training and test. (default: :obj:`s01.dat`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    if not os.path.exists(split_path):
        log.info(f'📊 | Create the split of train and test set.')
        log.info(
            f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
        )
        os.makedirs(split_path)
        info = dataset.info
        subjects = list(set(info['subject_id']))

        assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        subject_info = info[info['subject_id'] == subject]
        trial_ids = list(set(subject_info['trial_id']))

        train_info = None
        test_info = None

        for trial_id in trial_ids:
            cur_info = info[(info['subject_id'] == subject)
                            & (info['trial_id'] == trial_id)].reset_index()
            n_samples = len(cur_info)
            indices = np.arange(n_samples)

            train_index, test_index = model_selection.train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle)

            if train_info is None and test_info is None:
                train_info = [cur_info.iloc[train_index]]
                test_info = [cur_info.iloc[test_index]]
            else:
                train_info.append(cur_info.iloc[train_index])
                test_info.append(cur_info.iloc[test_index])

        train_info = pd.concat(train_info, ignore_index=True)
        test_info = pd.concat(test_info, ignore_index=True)

        train_info.to_csv(os.path.join(split_path, 'train.csv'), index=False)
        test_info.to_csv(os.path.join(split_path, 'test.csv'), index=False)

    else:
        log.info(
            f'📊 | Detected existing split of train and test set, use existing split from {split_path}.'
        )
        log.info(
            f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
        )

    train_info = pd.read_csv(os.path.join(split_path, 'train.csv'))
    test_info = pd.read_csv(os.path.join(split_path, 'test.csv'))

    train_dataset = copy(dataset)
    train_dataset.info = train_info

    test_dataset = copy(dataset)
    test_dataset.info = test_info

    return train_dataset, test_dataset



class Subcategory:
    r'''
    A tool class for separating out subsets of specified categories, often used to extract data for a certain type of paradigm, or for a certain type of task. Each subset in the formed subset list contains only one type of data.

    Common usage:

    .. code-block:: python

        from torcheeg.datasets import M3CVDataset
        from torcheeg.model_selection import Subcategory
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = Subcategory()
        dataset = M3CVDataset(root_path='./aistudio',
                              online_transform=transforms.Compose(
                                  [transforms.To2d(),
                                   transforms.ToTensor()]),
                              label_transform=transforms.Compose([
                                  transforms.Select('subject_id'),
                                  transforms.StringToInt()
                              ]))
        for subdataset in cv.split(dataset):
            loader = DataLoader(subdataset)
            ...

    TorchEEG supports the division of training and test sets within each subset after dividing the data into subsets. The sample code is as follows:

    .. code-block:: python

        cv = Subcategory()
        dataset = M3CVDataset(root_path='./aistudio',
                              online_transform=transforms.Compose(
                                  [transforms.To2d(),
                                   transforms.ToTensor()]),
                              label_transform=transforms.Compose([
                                  transforms.Select('subject_id'),
                                  transforms.StringToInt()
                              ]))
        for i, subdataset in enumerate(cv.split(dataset)):
            train_dataset, test_dataset = train_test_split(dataset=subdataset, split_path=f'./split{i}')

            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    For the already divided training and testing sets, TorchEEG recommends using two :obj:`Subcategory` to extract their subcategories respectively. On this basis, the :obj:`zip` function can be used to combine the subsets. It is worth noting that it is necessary to ensure that the training and test sets have the same number and variety of classes.

    .. code-block:: python

        train_cv = Subcategory()
        train_dataset = M3CVDataset(root_path='./aistudio',
                                    online_transform=transforms.Compose(
                                        [transforms.To2d(),
                                         transforms.ToTensor()]),
                                    label_transform=transforms.Compose([
                                        transforms.Select('subject_id'),
                                        transforms.StringToInt()
                                    ]))

        val_cv = Subcategory()
        val_dataset = M3CVDataset(root_path='./aistudio',
                                  subset='Calibration',
                                  num_channel=65,
                                  online_transform=transforms.Compose(
                                      [transforms.To2d(),
                                       transforms.ToTensor()]),
                                  label_transform=transforms.Compose([
                                      transforms.Select('subject_id'),
                                      transforms.StringToInt()
                                  ]))

        for train_dataset, val_dataset in zip(train_cv.split(train_dataset), val_cv.split(val_dataset)):
            train_loader = DataLoader(train_dataset)
            val_loader = DataLoader(val_dataset)
            ...

    Args:
        criteria (str): The classification criteria according to which we extract subsets of data for the including categories. (default: :obj:`'task'`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self, criteria: str = 'task', split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.criteria = criteria
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        assert self.criteria in list(
            info.columns
        ), f'Unsupported criteria {self.criteria}, please select one of the following options {list(info.columns)}.'

        category_list = list(set(info[self.criteria]))

        for category in category_list:
            subset_info = info[info[self.criteria] == category]
            subset_info.to_csv(os.path.join(self.split_path, f'{category}.csv'),
                               index=False)

    @property
    def category_list(self):
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'(.*).csv', indice_file)[0])

        category_list = list(set(map(indice_file_to_fold_id, indice_files)))
        category_list.sort()

        return category_list

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        category_list = self.category_list

        for category in category_list:
            subset_info = pd.read_csv(
                os.path.join(self.split_path, f'{category}.csv'))

            subset_dataset = copy(dataset)
            subset_dataset.info = subset_info

            yield subset_dataset

    @property
    def repr_body(self) -> Dict:
        return {'criteria': self.criteria, 'split_path': self.split_path}

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


class KFoldPerSubjectGroupbyTrial:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject dependent experiments. Experiments were performed separately for each subject, where the data for all trials of the subject is divided into k subsets at the trial dimension, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    .. image:: _static/KFoldPerSubjectGroupbyTrial.png
        :alt: The schematic diagram of KFoldPerSubjectGroupbyTrial
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubjectGroupbyTrial(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset):
            # The total number of experiments is the number subjects multiplied by K
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    :obj:`KFoldPerSubjectGroupbyTrial` allows the user to specify the index of the subject of interest, when the user need to report the performance on each subject.

    .. code-block:: python

        from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubjectGroupbyTrial(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset, subject=1):
            # k-fold cross-validation for subject 1
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[float, None] = None,
                 split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))
        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]

            subject_train_infos = {}
            subject_test_infos = {}

            trial_ids = list(set(subject_info['trial_id']))
            for trial_id in trial_ids:
                trial_info = subject_info[subject_info['trial_id'] == trial_id]

                for i, (train_index,
                        test_index) in enumerate(self.k_fold.split(trial_info)):
                    train_info = trial_info.iloc[train_index]
                    test_info = trial_info.iloc[test_index]

                    if not i in subject_train_infos:
                        subject_train_infos[i] = []

                    if not i in subject_test_infos:
                        subject_test_infos[i] = []

                    subject_train_infos[i].append(train_info)
                    subject_test_infos[i].append(test_info)

            for i in subject_train_infos.keys():
                subject_train_info = pd.concat(subject_train_infos[i],
                                               ignore_index=True)
                subject_test_info = pd.concat(subject_test_infos[i],
                                              ignore_index=True)
                subject_train_info.to_csv(os.path.join(
                    self.split_path, f'train_subject_{subject}_fold_{i}.csv'),
                    index=False)
                subject_test_info.to_csv(os.path.join(
                    self.split_path, f'test_subject_{subject}_fold_{i}.csv'),
                    index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    @property
    def fold_ids(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(
                re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][1])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(
            self,
            dataset: BaseDataset,
            subject: Union[int,
            None] = None) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        subjects = self.subjects
        fold_ids = self.fold_ids

        if not subject is None:
            assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if (not subject is None) and (local_subject != subject):
                continue

            for fold_id in fold_ids:
                train_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'train_subject_{local_subject}_fold_{fold_id}.csv'))
                test_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'test_subject_{local_subject}_fold_{fold_id}.csv'))

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                test_dataset = copy(dataset)
                test_dataset.info = test_info

                yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

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

class KFoldCrossTrial:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. One of the most commonly used data partitioning methods, where the data set is divided into k subsets of trials, with one subset trials being retained as the test set and the remaining k-1 subset trials being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    :obj:`KFoldCrossTrial` devides subsets at the dataset dimension. It means that during random sampling, adjacent signal samples may be assigned to the training set and the test set, respectively. When random sampling is not used, some subjects are not included in the training set. If you think these situations shouldn't happen, consider using :obj:`KFoldPerSubjectGroupbyTrial` or :obj:`KFoldGroupbyTrial`.

    .. image:: _static/KFoldCrossTrial.png
        :alt: The schematic diagram of KFoldCrossTrial
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldCrossTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = KFoldCrossTrial(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset):
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[None, int] = None,
                 split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        trial_ids = list(set(info['trial_id']))

        for fold_id, (train_index_trial_ids, test_index_trial_ids) in enumerate(
                self.k_fold.split(trial_ids)):

            if len(train_index_trial_ids) == 0 or len(
                    test_index_trial_ids) == 0:
                raise ValueError(
                    f'The number of training or testing trials is zero.')

            train_trial_ids = np.array(
                trial_ids)[train_index_trial_ids].tolist()
            test_trial_ids = np.array(trial_ids)[test_index_trial_ids].tolist()

            train_info = []
            for train_trial_id in train_trial_ids:
                train_info.append(info[info['trial_id'] == train_trial_id])
            train_info = pd.concat(train_info, ignore_index=True)

            test_info = []
            for test_trial_id in test_trial_ids:
                test_info.append(info[info['trial_id'] == test_trial_id])
            test_info = pd.concat(test_info, ignore_index=True)

            train_info.to_csv(os.path.join(self.split_path,
                                           f'train_fold_{fold_id}.csv'),
                              index=False)
            test_info.to_csv(os.path.join(self.split_path,
                                          f'test_fold_{fold_id}.csv'),
                             index=False)

    @property
    def fold_ids(self):
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        fold_ids = self.fold_ids

        for fold_id in fold_ids:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv'))
            test_info = pd.read_csv(
                os.path.join(self.split_path, f'test_fold_{fold_id}.csv'))

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

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
class KFoldGroupbyTrial:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. A variant of :obj:`KFold`, where the data set is divided into k subsets at the dimension of trials, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    :obj:`KFoldGroupbyTrial` devides subsets at the dimension of trials. Take the first partition with :obj:`k=5` as an example, the first 80% of samples of each trial are used for training, and the last 20% of samples are used for testing. It is more consistent with real applications and can test the generalization of the model to a certain extent.

    .. image:: _static/KFoldGroupbyTrial.png
        :alt: The schematic diagram of KFoldGroupbyTrial
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldGroupbyTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = KFoldGroupbyTrial(n_splits=5, shuffle=False)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset):
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[float, None] = None,
                 split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))

        train_infos = {}
        test_infos = {}

        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]

            trial_ids = list(set(subject_info['trial_id']))
            for trial_id in trial_ids:
                trial_info = subject_info[subject_info['trial_id'] == trial_id]
                for i, (train_index,
                        test_index) in enumerate(self.k_fold.split(trial_info)):
                    train_info = trial_info.iloc[train_index]
                    test_info = trial_info.iloc[test_index]

                    if not i in train_infos:
                        train_infos[i] = []

                    if not i in test_infos:
                        test_infos[i] = []

                    train_infos[i].append(train_info)
                    test_infos[i].append(test_info)

        for i in train_infos.keys():
            train_info = pd.concat(train_infos[i], ignore_index=True)
            test_info = pd.concat(test_infos[i], ignore_index=True)
            train_info.to_csv(os.path.join(self.split_path,
                                           f'train_fold_{i}.csv'),
                              index=False)
            test_info.to_csv(os.path.join(self.split_path,
                                          f'test_fold_{i}.csv'),
                             index=False)

    @property
    def fold_ids(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        fold_ids = self.fold_ids

        for fold_id in fold_ids:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv'))
            test_info = pd.read_csv(
                os.path.join(self.split_path, f'test_fold_{fold_id}.csv'))

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

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
class KFoldCrossSubject:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. One of the most commonly used data partitioning methods, where the data set is divided into k subsets of subjects, with one subset subjects being retained as the test set and the remaining k-1 subset subjects being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    .. image:: _static/KFoldCrossSubject.png
        :alt: The schematic diagram of KFoldCrossSubject
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldCrossSubject
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = KFoldCrossSubject(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset):
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[None, int] = None,
                 split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subject_ids = list(set(info['subject_id']))

        for fold_id, (train_index_subject_ids,
                      test_index_subject_ids) in enumerate(
                          self.k_fold.split(subject_ids)):

            if len(train_index_subject_ids) == 0 or len(
                    test_index_subject_ids) == 0:
                raise ValueError(
                    f'The number of training or testing subjects is zero.')

            train_subject_ids = np.array(
                subject_ids)[train_index_subject_ids].tolist()
            test_subject_ids = np.array(
                subject_ids)[test_index_subject_ids].tolist()

            train_info = []
            for train_subject_id in train_subject_ids:
                train_info.append(info[info['subject_id'] == train_subject_id])
            train_info = pd.concat(train_info, ignore_index=True)

            test_info = []
            for test_subject_id in test_subject_ids:
                test_info.append(info[info['subject_id'] == test_subject_id])
            test_info = pd.concat(test_info, ignore_index=True)

            train_info.to_csv(os.path.join(self.split_path,
                                           f'train_fold_{fold_id}.csv'),
                              index=False)
            test_info.to_csv(os.path.join(self.split_path,
                                          f'test_fold_{fold_id}.csv'),
                             index=False)

    @property
    def fold_ids(self):
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        fold_ids = self.fold_ids

        for fold_id in fold_ids:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv'))
            test_info = pd.read_csv(
                os.path.join(self.split_path, f'test_fold_{fold_id}.csv'))

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

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

class KFoldPerSubject:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject dependent experiments. Experiments were performed separately for each subject, where the data of the subject is divided into k subsets, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    .. image:: _static/KFoldPerSubject.png
        :alt: The schematic diagram of KFoldPerSubject
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.model_selection import KFoldPerSubject
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubject(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset):
            # The total number of experiments is the number subjects multiplied by K
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    :obj:`KFoldPerSubject` allows the user to specify the index of the subject of interest, when the user need to report the performance on each subject.

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.model_selection import KFoldPerSubject
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubject(n_splits=5, shuffle=True)
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        for train_dataset, test_dataset in cv.split(dataset, subject=1):
            # k-fold cross-validation for subject 1
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[float, None] = None,
                 split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))
        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]

            for i, (train_index,
                    test_index) in enumerate(self.k_fold.split(subject_info)):
                train_info = subject_info.iloc[train_index]
                test_info = subject_info.iloc[test_index]

                train_info.to_csv(os.path.join(
                    self.split_path, f'train_subject_{subject}_fold_{i}.csv'),
                    index=False)
                test_info.to_csv(os.path.join(
                    self.split_path, f'test_subject_{subject}_fold_{i}.csv'),
                    index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    @property
    def fold_ids(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(
                re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][1])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(
            self,
            dataset: BaseDataset,
            subject: Union[int,
            None] = None) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        subjects = self.subjects
        fold_ids = self.fold_ids

        if not subject is None:
            assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if (not subject is None) and (local_subject != subject):
                continue

            for fold_id in fold_ids:
                train_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'train_subject_{local_subject}_fold_{fold_id}.csv'))
                test_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'test_subject_{local_subject}_fold_{fold_id}.csv'))

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                test_dataset = copy(dataset)
                test_dataset.info = test_info

                yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

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