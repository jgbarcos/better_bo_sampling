import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import numpy as np

def append_colorbar(ax, fig, z, size='7%', pad='2%', cmap_name='viridis', zero_start=False):
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size=size, pad=pad)

    vmin = np.nanmin(z)
    vmax = np.nanmax(z)
    
    if zero_start:
        vmin = 0
    
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(norm, cmap=cmap_name)
    sm.set_array(z)
    fig.colorbar(sm, cax)

def _plot_axis_aligned_box(ax, xmin, xmax, ymin, ymax, **kwargs):
    corners_x = [xmin, xmax, xmax, xmin, xmin]
    corners_y = [ymin, ymin, ymax, ymax, ymin]
    for i in range(len(corners_x)):
        ax.plot(corners_x[i:i+2], corners_y[i:i+2], **kwargs)

def plot_2d_bounds(ax, bounds, border=1.0, strict_limit=False):
    xmin, xmax, ymin, ymax = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]

    # Set a margin and mark with a red border the function bounds
    if border > 1.0:
        mid_x = 0.5 * (xmin + xmax)
        mid_y = 0.5 * (ymin + ymax)
        rad_x = border * 0.5 * (xmax - xmin)
        rad_y = border * 0.5 * (ymax - ymin)

        if strict_limit:
            ax.set_xlim(xmin=mid_x-rad_x, xmax=mid_x+rad_x)
            ax.set_ylim(ymin=mid_y-rad_y, ymax=mid_y+rad_y)
        else:
            # Hack: transparent box to force at least BORDER_BOUNDS
            _plot_axis_aligned_box(ax, mid_x-rad_x, mid_x+rad_x, mid_y-rad_y, mid_y+rad_y, alpha=0.0)

    # Set axis bounds to function bounds
    elif strict_limit:
        ax.set_xlim(xmin=bounds[0][0], xmax=bounds[0][1])
        ax.set_ylim(ymin=bounds[1][0], ymax=bounds[1][1])

    # Plot a red box on the bounds
    _plot_axis_aligned_box(ax, xmin, xmax, ymin, ymax, c='r', marker='', linewidth=0.3)