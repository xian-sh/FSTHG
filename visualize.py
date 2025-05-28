import io
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import PIL
import torch
from matplotlib import colors
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
from pylab import cm

mne.set_log_level('CRITICAL')

default_montage = mne.channels.make_standard_montage('standard_1020')


def plot2image(ploter, save_path=None):
    if save_path:
        # 直接保存为PNG文件
        ploter.savefig(save_path, format='png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        return None
    else:
        # 原来的方式，返回PIL Image对象
        buf = io.BytesIO()
        ploter.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        buf.seek(0)
        return PIL.Image.open(buf)


def plot_raw_topomap(tensor: torch.Tensor,
                     channel_list: List[str],
                     sampling_rate: int,
                     plot_second_list: List[int] = [0.0, 0.25, 0.5, 0.75],
                     montage: mne.channels.DigMontage = default_montage):
    r'''
    Plot a topographic map of the input raw EEG signal as image.

    .. code-block:: python

        from torcheeg.utils import plot_raw_topomap
        from torcheeg.constants import DEAP_CHANNEL_LIST

        eeg = torch.randn(32, 128)
        img = plot_raw_topomap(eeg,
                         channel_list=DEAP_CHANNEL_LIST,
                         sampling_rate=128)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_raw_topomap.png
        :alt: The output image of plot_raw_topomap
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input EEG signal, the shape should be [number of channels, number of data points].
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        sampling_rate (int): Sample rate of the data.
        plot_second_list (list): The time (second) at which the topographic map is drawn. (default: :obj:`[0.0, 0.25, 0.5, 0.75]`)
        montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ch_types = ['eeg'] * len(channel_list)
    info = mne.create_info(ch_names=channel_list,
                           ch_types=ch_types,
                           sfreq=sampling_rate)
    tensor = tensor.detach().cpu().numpy()
    info.set_montage(montage,
                     match_alias=True,
                     match_case=False,
                     on_missing='ignore')
    fig, axes = plt.subplots(1, len(plot_second_list), figsize=(20, 5))
    for i, label in enumerate(plot_second_list):
        mne.viz.plot_topomap(tensor[:, int(sampling_rate * label)],
                             info,
                             axes=axes[i],
                             show=False,
                             sphere=(0., 0., 0., 0.11))
        axes[i].set_title(f'{label}s', {
            'fontsize': 24,
            'fontname': 'Liberation Serif'
        })

    img = plot2image(fig)
    plt.show()
    return np.array(img)


def plot_feature_topomap(tensor: torch.Tensor,
                         channel_list: List[str],
                         feature_list: Union[List[str], None] = None,
                         montage: mne.channels.DigMontage = default_montage,
                         fig_shape: Tuple[int, int] = None):
    r'''
    Plot a topographic map of the input EEG features as image.

    .. code-block:: python

        from torcheeg.utils import plot_feature_topomap
        from torcheeg.constants import DEAP_CHANNEL_LIST

        eeg = torch.randn(32, 4)
        img = plot_feature_topomap(eeg,
                         channel_list=DEAP_CHANNEL_LIST,
                         feature_list=["theta", "alpha", "beta", "gamma"])
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_feature_topomap.png
        :alt: The output image of plot_feature_topomap
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input EEG signal, the shape should be [number of channels, dimensions of features].
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        feature_list (list): . The names of feature dimensions displayed on the output image, whose length should be consistent with the dimensions of features. If set to None, the dimension index of the feature is used instead. (default: :obj:`None`)
        montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
        fig_shape (Tuple[int, int], optional): The shape of the sub graphs (width, height). If `None`, the layout is automatically set to (1, len(feature_list)). (default: :obj:`None`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ch_types = ['eeg'] * len(channel_list)
    info = mne.create_info(ch_names=channel_list, ch_types=ch_types, sfreq=128)
    tensor = tensor.detach().cpu().numpy()
    info.set_montage(montage,
                     match_alias=True,
                     match_case=False,
                     on_missing='ignore')

    if feature_list is None:
        feature_list = list(range(tensor.shape[1]))
    num_subplots = len(feature_list)

    if fig_shape == None:
        fig_shape = (1, num_subplots)
    else:
        if len(fig_shape) != 2:
            raise ValueError(
                "fig_shape only support 2d graph, so just contain width and height"
            )
        if not all(isinstance(n, int) and n > 0 for n in fig_shape):
            raise ValueError(
                "width and height in fig_shape must be positive integers")
        if fig_shape[0] * fig_shape[1] != num_subplots:
            raise ValueError(
                f"The product of width and height in fig_shape must equal feature_list length: {num_subplots}"
            )

    fig, axes = plt.subplots(fig_shape[0],
                             fig_shape[1],
                             figsize=(fig_shape[1] * 5, fig_shape[0] * 5),
                             squeeze=False)

    if num_subplots > 1:
        for i, (label) in enumerate(feature_list):
            row, col = i // fig_shape[1], i % fig_shape[1]
            mne.viz.plot_topomap(tensor[:, i],
                                 info,
                                 axes=axes[row, col],
                                 show=False,
                                 sphere=(0., 0., 0., 0.11))
            axes[row, col].set_title(label, {
                'fontsize': 24,
                'fontname': 'Liberation Serif'
            })
    else:
        mne.viz.plot_topomap(tensor[:, 0],
                             info,
                             axes=axes,
                             show=False,
                             sphere=(0., 0., 0., 0.11))
        axes.set_title(feature_list[0], {
            'fontsize': 24,
            'fontname': 'Liberation Serif'
        })

    img = plot2image(fig)
    plt.show()
    return np.array(img)


def plot_signal(tensor: torch.Tensor,
                channel_list: List[str],
                sampling_rate: int,
                montage: mne.channels.DigMontage = default_montage):
    r'''
    Plot signal values of the input raw EEG as image.

    .. code-block:: python

        import torch

        from torcheeg.utils import plot_signal
        from torcheeg.constants import DEAP_CHANNEL_LIST

        eeg = torch.randn(32, 128)
        img = plot_signal(eeg,
                          channel_list=DEAP_CHANNEL_LIST,
                          sampling_rate=128)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_signal.png
        :alt: The output image of plot_signal
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input EEG signal, the shape should be [number of channels, number of data points].
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        sampling_rate (int): Sample rate of the data.
        montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ch_types = ['misc'] * len(channel_list)
    info = mne.create_info(ch_names=channel_list,
                           ch_types=ch_types,
                           sfreq=sampling_rate)

    epochs = mne.io.RawArray(tensor.detach().cpu().numpy(), info)
    epochs.set_montage(montage, match_alias=True, on_missing='ignore')
    img = plot2image(
        epochs.plot(show_scrollbars=False, show_scalebars=False, block=True))
    plt.show()
    return np.array(img)


def plot_3d_tensor(tensor: torch.Tensor,
                   color: Union[colors.Colormap, str] = 'hsv',
                   location_list: List[List[str]] = None,
                   figsize: Tuple = (8, 6),
                   save_path: str = './'):
    r'''
    Visualize a 3-D matrices in 3-D space with transparent spots for '-' positions.

    Args:
        tensor (torch.Tensor): The input 3-D tensor.
        color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)
        location_list (List[List[str]]): The electrode location list. (default: :obj:`None`)
        figsize (Tuple): Figure size. (default: :obj:`(8, 6)`)
        save_path (str): Path to save the output image. (default: :obj:`'./'`)

    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ndarray = tensor.detach().cpu().numpy()

    # Create alpha mask (0 for transparent, 1 for visible)
    filled = np.ones_like(ndarray, dtype=bool)
    if location_list is not None:
        for i in range(len(location_list)):
            for j in range(len(location_list[i])):
                if location_list[i][j] == '-':
                    filled[:, i, j] = False

    # Normalize the data to [0, 1]
    ndarray = (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())

    # Create colormap
    colormap = cm.get_cmap(color)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')

    # Plot voxels
    voxels = ax.voxels(filled,
                       facecolors=colormap(ndarray),
                       edgecolors='k',
                       linewidths=0.1,
                       shade=False)

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.axis('off')
    # Adjust grid line properties if needed
    ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)  # 设置网格线颜色和透明度
    ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)
    ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)

    # Adjust the view to face front-right
    ax.view_init(elev=25, azim=45, )
    # ax.view_init(elev=135, azim=25, )

    plt.tight_layout()

    # Save and show the plot
    img = plot2image(fig, save_path)
    plt.show()

    return np.array(img)


# def plot_3d_tensor(tensor: torch.Tensor,
#                    color: Union[colors.Colormap, str] = 'hsv'):
#     r'''
#     Visualize a 3-D matrices in 3-D space.
#
#     .. code-block:: python
#
#         from torcheeg.utils import plot_3d_tensor
#
#         eeg = torch.randn(128, 9, 9)
#         img = plot_3d_tensor(eeg)
#         # If using jupyter, the output image will be drawn on notebooks.
#
#     .. image:: _static/plot_3d_tensor.png
#         :alt: The output image of plot_3d_tensor
#         :align: center
#
#     |
#
#     Args:
#         tensor (torch.Tensor): The input 3-D tensor.
#         color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)
#
#     Returns:
#         np.ndarray: The output image in the form of :obj:`np.ndarray`.
#     '''
#     ndarray = tensor.detach().cpu().numpy()
#
#     filled = np.ones_like(ndarray, dtype=bool)
#     colormap = cm.get_cmap(color)
#
#     ndarray = (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())
#
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.voxels(filled,
#               facecolors=colormap(ndarray),
#               edgecolors='k',
#               linewidths=0.1,
#               shade=False)
#
#     img = plot2image(fig)
#     plt.show()
#
#     return np.array(img)


# def plot_2d_tensor(tensor: torch.Tensor,
#                    color: Union[colors.Colormap, str] = 'ReGn'):  # hsv
#     r'''
#     Visualize a 2-D matrices in 2-D space.
#
#     .. code-block:: python
#
#         import torch
#
#         from torcheeg.utils import plot_2d_tensor
#
#         eeg = torch.randn(9, 9)
#         img = plot_2d_tensor(eeg)
#         # If using jupyter, the output image will be drawn on notebooks.
#
#     .. image:: _static/plot_2d_tensor.png
#         :alt: The output image of plot_2d_tensor
#         :align: center
#
#     |
#
#     Args:
#         tensor (torch.Tensor): The input 2-D tensor.
#         color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)
#
#     Returns:
#         np.ndarray: The output image in the form of :obj:`np.ndarray`.
#     '''
#     ndarray = tensor.detach().cpu().numpy()
#
#     fig = plt.figure()
#     ax = plt.axes()
#
#     colormap = cm.get_cmap(color)
#     ax.imshow(ndarray, cmap=colormap, interpolation='nearest')
#
#     img = plot2image(fig)
#     plt.show()
#
#     return np.array(img)

def plot_2d_tensor(tensor: torch.Tensor,
                   color: Union[colors.Colormap, str] = 'RdBu',
                   location_list: List[List[str]] = None,
                   figsize: Tuple = (8,6),
                   fontsize: int = 4,
                   save_path: str = './'):
    r'''
    Visualize a 2-D matrices in 2-D space with transparent spots for '-' positions.

    Args:
        tensor (torch.Tensor): The input 2-D tensor.
        color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`RdBu`)
        location_list (List[List[str]]): The electrode location list. (default: :obj:`DEAP_LOCATION_LIST`)

    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ndarray = tensor.squeeze(0).detach().cpu().numpy()

    # Create alpha mask (0 for transparent, 1 for visible)
    alpha_mask = np.ones_like(ndarray)
    for i in range(len(location_list)):
        for j in range(len(location_list[i])):
            if location_list[i][j] == '-':
                alpha_mask[i, j] = 0

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    # Plot with alpha mask
    colormap = cm.get_cmap(color)
    im = ax.imshow(ndarray, cmap=colormap, interpolation='nearest', alpha=alpha_mask)

    # Add colorbar
    # plt.colorbar(im)

    # Add electrode labels
    for i in range(len(location_list)):
        for j in range(len(location_list[i])):
            if location_list[i][j] != '-':
                ax.text(j, i, location_list[i][j],
                        ha='center', va='center',
                        color='black', fontsize=fontsize, fontweight='bold')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Set background to white
    ax.set_facecolor('white')

    plt.tight_layout()

    img = plot2image(fig, save_path)
    plt.show()

    return np.array(img)

def plot_adj_connectivity(adj: torch.Tensor,
                          channel_list: list = None,
                          region_list: list = None,
                          num_connectivity: int = 60,
                          linewidth: float = 1.5):
    r'''
    Visualize connectivity between nodes in an adjacency matrix, using circular networks.

    .. code-block:: python

        import torch
        
        from torcheeg.utils import plot_adj_connectivity
        from torcheeg.constants import SEED_CHANNEL_LIST
        
        adj = torch.randn(62, 62) # relationship between 62 electrodes
        img = plot_adj_connectivity(adj, SEED_CHANNEL_LIST)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_adj_connectivity.png
        :alt: The output image of plot_adj_connectivity
        :align: center

    |
    
    Args:
        adj (torch.Tensor): The input 2-D adjacency tensor.
        channel_list (list): The electrode name of the row/column in the input adjacency matrix, used to label the electrode corresponding to the node on circular networks. If set to None, the electrode's index is used. (default: :obj:`None`)
        region_list (list): region_list (list): The region list where the electrodes are divided into different brain regions. If set, electrodes in the same area will be aligned on the map and filled with the same color. (default: :obj:`None`)
        num_connectivity (int): The number of connections to retain on circular networks, where edges with larger weights in the adjacency matrix will be limitedly retained, and the excess is omitted. (default: :obj:`50`)
        linewidth (float): Line width to use for connections. (default: :obj:`1.5`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    if channel_list is None:
        channel_list = list(range(len(adj)))
    adj = adj.detach().cpu().numpy()
    assert len(channel_list) == adj.shape[0] and len(channel_list) == adj.shape[
        1], 'The size of the adjacency matrix does not match the number of channel names.'

    node_colors = None
    if region_list:
        num_region = len(region_list)
        colormap = matplotlib.cm.get_cmap('rainbow')
        region_colors = list(colormap(np.linspace(0, 1, num_region)))
        # random.shuffle(region_colors)
        # # circle two colors is better
        # colors = colormap(np.linspace(0, 1, 4))
        # region_colors = [
        #     colors[1] if region_index % 2 == 0 else colors[2]
        #     for region_index in range(num_region)
        # ]

        new_channel_list = []
        new_adj_order = []
        for region_index, region in enumerate(region_list):
            for electrode_index in region:
                new_adj_order.append(electrode_index)
                new_channel_list.append(channel_list[electrode_index])
        new_adj = adj[new_adj_order][:, new_adj_order]

        electrode_colors = [None] * len(new_channel_list)
        i = 0
        for region_index, region in enumerate(region_list):
            for electrode_index in region:
                electrode_colors[i] = region_colors[region_index]
                i += 1

        adj = new_adj
        channel_list = new_channel_list
        node_colors = electrode_colors

    node_angles = circular_layout(channel_list, channel_list, start_pos=90)
    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    fig, ax = plt.subplots(figsize=(2, 2),
                           facecolor='white',
                           subplot_kw=dict(polar=True))
    plot_connectivity_circle(adj,
                             channel_list,
                             node_colors=node_colors,
                             n_lines=num_connectivity,
                             node_angles=node_angles,
                             ax=ax,
                             facecolor='white',
                             textcolor='black',
                             node_edgecolor='white',
                             colormap='autumn',
                             colorbar=False,
                             padding=0.0,
                             linewidth=linewidth,
                             fontsize_names=10)
    fig.tight_layout()
    img = plot2image(fig)
    plt.show()

    return np.array(img)
