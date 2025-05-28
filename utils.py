import os
import re
from copy import copy
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from itertools import chain

import logging
import datetime
import sys
import random

from datasets.base import BaseDataset
from datasets.utils import get_random_dir_path

log = logging.getLogger('torcheeg')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多张 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    # d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

def set_logging_config(logdir):
    """
    Logging configuration
    :param logdir: Directory to save log files
    :return: None
    """
    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.Formatter.converter = beijing

    logging.basicConfig(
        format="[%(asctime)s] [%(name)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(logdir, "log.txt"), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


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

class Subcategory_deap:
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
            subset_info.to_csv(os.path.join(self.split_path, f's{category:02d}.dat.csv'),
                               index=False)

    @property
    def category_list(self):
        indice_files = list(os.listdir(self.split_path))

        # def indice_file_to_fold_id(indice_file):
        #     return int(re.findall(r'(.*).csv', indice_file)[0])
        def indice_file_to_fold_id(indice_file):
            # Extract number from 's01.dat', 's02.dat', etc.
            match = re.findall(r's(\d+)\.dat\.csv', indice_file)
            if match:
                return int(match[0])
            raise ValueError(f"Unexpected filename format: {indice_file}")

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
                os.path.join(self.split_path, f's{category:02d}.dat.csv'))

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


class LeaveOneSubjectOut:
    r'''
    A tool class for leave-one-subject-out cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject independent experiments. During each fold, experiments require testing on one subject and training on the other subjects.

    .. image:: _static/LeaveOneSubjectOut.png
        :alt: The schematic diagram of LeaveOneSubjectOut
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import LeaveOneSubjectOut
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = LeaveOneSubjectOut()
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
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self, split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))

        for test_subject in subjects:
            train_subjects = subjects.copy()
            train_subjects.remove(test_subject)

            train_info = []
            for train_subject in train_subjects:
                train_info.append(info[info['subject_id'] == train_subject])

            train_info = pd.concat(train_info)
            test_info = info[info['subject_id'] == test_subject]

            train_info.to_csv(os.path.join(self.split_path,
                                           f'train_subject_{test_subject}.csv'),
                              index=False)
            test_info.to_csv(os.path.join(self.split_path,
                                          f'test_subject_{test_subject}.csv'),
                             index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*).csv', indice_file)[0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

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

        subjects = self.subjects

        for subject in subjects:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_subject_{subject}.csv'))
            test_info = pd.read_csv(
                os.path.join(self.split_path, f'test_subject_{subject}.csv'))

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, test_dataset


_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader


def classification_metrics(metric_list: List[str], num_classes: int):
    allowed_metrics = [
        'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc',
        'kappa'
    ]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', 'kappa'."
            )
    metric_dict = {
        'accuracy':
            torchmetrics.Accuracy(task='multiclass',
                                  num_classes=num_classes,
                                  top_k=1),
        'precision':
            torchmetrics.Precision(task='multiclass',
                                   average='macro',
                                   num_classes=num_classes),
        'recall':
            torchmetrics.Recall(task='multiclass',
                                average='macro',
                                num_classes=num_classes),
        'f1score':
            torchmetrics.F1Score(task='multiclass',
                                 average='macro',
                                 num_classes=num_classes),
        'matthews':
            torchmetrics.MatthewsCorrCoef(task='multiclass',
                                          num_classes=num_classes),
        'auroc':
            torchmetrics.AUROC(task='multiclass', num_classes=num_classes),
        'kappa':
            torchmetrics.CohenKappa(task='multiclass', num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG classification.

        .. code-block:: python

            trainer = ClassifierTrainer(model)
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
            num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', and 'kappa'. (default: :obj:`["accuracy"]`)

        .. automethod:: fit
        .. automethod:: test
    '''

    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super().__init__()
        self.model = model

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.CrossEntropyLoss()

        self.init_metrics(metrics, num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(trainable_parameters,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        return y_hat

class DualDataLoader:

    def __init__(self, ref_dataloader: DataLoader,
                 other_dataloader: DataLoader):
        self.ref_dataloader = ref_dataloader
        self.other_dataloader = other_dataloader

    def __iter__(self):
        return self.dual_iterator()

    def __len__(self):
        return len(self.ref_dataloader)

    def dual_iterator(self):
        other_it = iter(self.other_dataloader)
        for data in self.ref_dataloader:
            try:
                data_ = next(other_it)
            except StopIteration:
                other_it = iter(self.other_dataloader)
                data_ = next(other_it)
            yield data, data_


class _MMDLikeTrainer(ClassifierTrainer):

    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 num_classes: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 weight_domain: float = 1.0,
                 weight_scheduler: bool = True,
                 lr_scheduler_gamma: float = 0.0,
                 lr_scheduler_decay: float = 0.75,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):
        super(ClassifierTrainer, self).__init__()

        self.extractor = extractor
        self.classifier = classifier

        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_domain = weight_domain
        self.weight_scheduler = weight_scheduler

        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.lr_scheduler_decay = lr_scheduler_decay
        self.lr_scheduler = not lr_scheduler_gamma == 0.0
        self.warmup_epochs = warmup_epochs

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.num_classes = num_classes
        self.metrics = metrics
        self.init_metrics(metrics, num_classes)

        self.ce_fn = nn.CrossEntropyLoss()

        self.num_batches = None  # init in 'fit' method
        self.non_warmup_epochs = None  # init in 'fit' method
        self.lr_factor = 1.0  # update in 'on_train_batch_start' method
        self.weight_factor = 1.0  # update in 'on_train_batch_start' method
        self.scheduled_weight_domain = 1.0  # update in 'on_train_batch_start' method

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_domain_loss = torchmetrics.MeanMetric()
        self.train_task_loss = torchmetrics.MeanMetric()

        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(self,
            source_loader: DataLoader,
            target_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs):
        r'''
        Args:
            source_loader (DataLoader): Iterable DataLoader for traversing the data batch from the source domain (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            target_loader (DataLoader): Iterable DataLoader for traversing the training data batch from the target domain (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc). The target dataset does not have to return labels.
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): The maximum number of epochs to train. (default: :obj:`300`)
        '''

        train_loader = DualDataLoader(source_loader, target_loader)

        self.num_batches = len(train_loader)
        self.non_warmup_epochs = max_epochs - self.warmup_epochs

        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def on_train_batch_start(self, batch, batch_idx):
        if self.current_epoch >= self.warmup_epochs:
            delta_epoch = self.current_epoch - self.warmup_epochs
            p = (batch_idx + delta_epoch * self.num_batches) / (
                self.non_warmup_epochs * self.num_batches)
            self.weight_factor = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            if self.lr_scheduler:
                self.lr_factor = 1.0 / ((1.0 + self.lr_scheduler_gamma * p)**
                                        self.lr_scheduler_decay)

        if self.weight_scheduler:
            self.scheduled_weight_domain = self.weight_domain * self.weight_factor

    def _domain_loss_fn(self, x_source_feat: torch.Tensor,
                        x_target_feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        (x_source, y_source), (x_target, _) = batch

        x_source_feat = self.extractor(x_source)
        y_source_pred = self.classifier(x_source_feat)
        x_target_feat = self.extractor(x_target)

        domain_loss = self._domain_loss_fn(x_source_feat, x_target_feat)

        task_loss = self.ce_fn(y_source_pred, y_source)

        if self.current_epoch >= self.warmup_epochs:
            loss = task_loss + self.scheduled_weight_domain * domain_loss
        else:
            loss = task_loss

        self.log("train_domain_loss",
                 self.train_domain_loss(domain_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_task_loss",
                 self.train_task_loss(task_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_source_pred, y_source),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_domain_loss",
                 self.train_domain_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("train_task_loss",
                 self.train_task_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.train_domain_loss.reset()
        self.train_task_loss.reset()
        self.train_metrics.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{
            "params": self.extractor.parameters()
        }, {
            "params": self.classifier.parameters(),
            "lr": 10 * self.lr
        }],
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: self.lr_factor)
            return [optimizer], [scheduler]
        return [optimizer]


class WalkerLoss(nn.Module):
    def forward(self, P_aba, y):
        equality_matrix = torch.eq(y.reshape(-1, 1), y).float()
        p_target = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
        p_target.requires_grad = False

        L_walker = F.kl_div(torch.log(1e-8 + P_aba),
                            p_target, reduction='sum')
        L_walker /= p_target.shape[0]

        return L_walker


class VisitLoss(nn.Module):
    def forward(self, P_b):
        p_visit = torch.ones([1, P_b.shape[1]]) / float(P_b.shape[1])
        p_visit.requires_grad = False
        p_visit = p_visit.to(P_b.device)
        L_visit = F.kl_div(torch.log(1e-8 + P_b), p_visit, reduction='sum')
        L_visit /= p_visit.shape[0]

        return L_visit


class AssociationMatrix(nn.Module):
    def __init__(self):
        super(AssociationMatrix, self).__init__()

    def forward(self, X_source, X_target):
        X_source = X_source.reshape(X_source.shape[0], -1)
        X_target = X_target.reshape(X_target.shape[0], -1)

        W = torch.mm(X_source, X_target.transpose(1, 0))

        P_ab = F.softmax(W, dim=1)
        P_ba = F.softmax(W.transpose(1, 0), dim=1)

        P_aba = P_ab.mm(P_ba)
        P_b = torch.mean(P_ab, dim=0, keepdim=True)

        return P_aba, P_b


class AssociativeLoss(nn.Module):
    def __init__(self, walker_weight=1., visit_weight=1.):
        super(AssociativeLoss, self).__init__()

        self.matrix = AssociationMatrix()
        self.walker = WalkerLoss()
        self.visit = VisitLoss()

        self.walker_weight = walker_weight
        self.visit_weight = visit_weight

    def forward(self, X_source, X_target, y):

        P_aba, P_b = self.matrix(X_source, X_target)
        L_walker = self.walker(P_aba, y)
        L_visit = self.visit(P_b)

        return self.visit_weight * L_visit + self.walker_weight * L_walker


class ADATrainer(_MMDLikeTrainer):
    r'''
    This class supports the implementation of Associative Domain Adaptation (ADA) for deep domain adaptation.

    NOTE: ADA belongs to unsupervised domain adaptation methods, which only use labeled source data and unlabeled target data. This means that the target dataset does not have to contain labels.

    - Paper: Haeusser P, Frerix T, Mordvintsev A, et al. Associative domain adaptation[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2765-2773.
    - URL: https://arxiv.org/abs/1708.00938
    - Related Project: https://github.com/stes/torch-assoc

    .. code-block:: python

        from torcheeg.models import CCNN
        from torcheeg.trainers import ADATrainer

        class Extractor(CCNN):
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = x.flatten(start_dim=1)
                return x

        class Classifier(CCNN):
            def forward(self, x):
                x = self.lin1(x)
                x = self.lin2(x)
                return x

        extractor = Extractor(in_channels=5, num_classes=3)
        classifier = Classifier(in_channels=5, num_classes=3)

        trainer = ADATrainer(extractor,
                             classifier,
                             num_classes=3,
                             devices=1,
                             weight_visit=0.6,
                             accelerator='gpu')

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        classifier (nn.Module): The classification model learns the classification task with the source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        num_classes (int, optional): The number of categories in the dataset. (default: :obj:`None`)
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        weight_walker (float): The weight of the walker loss. (default: :obj:`1.0`)
        weight_visit (float): The weight of the visit loss. (default: :obj:`1.0`)
        weight_domain (float): The weight of the associative loss (default: :obj:`1.0`)
        weight_scheduler (bool): Whether to use a scheduler for the weight of the associative loss, growing from 0 to 1 following the schedule from the DANN paper. (default: :obj:`False`)
        lr_scheduler (bool): Whether to use a scheduler for the learning rate, as defined in the DANN paper. (default: :obj:`False`)
        warmup_epochs (int): The number of epochs for the warmup phase, during which the weight of the associative loss is 0. (default: :obj:`0`)
        devices (int): The number of devices to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', and 'kappa'. (default: :obj:`["accuracy"]`)

    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 num_classes: int,
                 lr: float = 1e-4,
                 weight_walker: float = 1.0,
                 weight_visit: float = 1.0,
                 weight_domain: float = 1.0,
                 weight_decay: float = 0.0,
                 weight_scheduler: bool = True,
                 lr_scheduler_gamma: float = 0.0,
                 lr_scheduler_decay: float = 0.75,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super(ADATrainer,
              self).__init__(extractor=extractor,
                             classifier=classifier,
                             num_classes=num_classes,
                             lr=lr,
                             weight_decay=weight_decay,
                             weight_domain=weight_domain,
                             weight_scheduler=weight_scheduler,
                             lr_scheduler_gamma=lr_scheduler_gamma,
                             lr_scheduler_decay=lr_scheduler_decay,
                             warmup_epochs=warmup_epochs,
                             devices=devices,
                             accelerator=accelerator,
                             metrics=metrics)
        self.weight_walker = weight_walker
        self.weight_visit = weight_visit

        self._assoc_fn = AssociativeLoss(weight_walker, weight_visit)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        (x_source, y_source), (x_target, _) = batch

        x_source_feat = self.extractor(x_source)
        y_source_pred = self.classifier(x_source_feat)
        x_target_feat = self.extractor(x_target)

        domain_loss = self._domain_loss_fn(x_source_feat, x_target_feat,
                                           y_source)

        task_loss = self.ce_fn(y_source_pred, y_source)
        if self.current_epoch >= self.warmup_epochs:
            loss = task_loss + self.scheduled_weight_domain * domain_loss
        else:
            loss = task_loss

        self.log("train_domain_loss",
                 self.train_domain_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_task_loss",
                 self.train_task_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_source_pred, y_source),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def _domain_loss_fn(self, x_source_feat: torch.Tensor,
                        x_target_feat: torch.Tensor,
                        y_source: torch.Tensor) -> torch.Tensor:

        return self._assoc_fn(x_source_feat, x_target_feat, y_source)


