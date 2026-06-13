# module: plot_maps.py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.transforms import Affine2D
import numpy as np

from maps import arcsecond


def sanitize_log_data(data):
    '''
    Sanitize data before plotting with LogNorm.

    Replaces non-finite values (NaN or inf) with zeros and clips the
    minimum value to a small positive threshold to avoid issues with
    logarithmic normalization.

    Parameters:
    -----------
    data : array-like
        Input array containing emission map values.

    Returns:
    --------
    ndarray
        Sanitized array with only finite positive values.
    '''

    data = np.asarray(data)
    data = np.where(np.isfinite(data), data, 0.0)

    return np.clip(data, 1e-300, None)


def get_norm(data):
    '''
    Logarithmic normalization for emission maps.

    The normalization range is defined between
    vmax / 500 and 1.1 * vmax.

    Parameters:
    -----------
    data : ndarray
        Intensity map.

    Returns:
    --------
    Logarithmic normalization object for plotting.
    '''

    data = data[np.isfinite(data) & (data > 0)]

    if len(data) == 0:
        return LogNorm(vmin=1e-30, vmax=1e-20)

    vmax = np.max(data)

    vmin = vmax / 100.

    return LogNorm(vmin=vmin, vmax=1.1 * vmax)


def compute_R0_position(inclination, distance, R0_corrected):
    '''
    Compute the projected position of the apex on the sky.

    Calculated from stellar position
    and rotated according to the inclination and position angle.

    Parameters:
    -----------
    inclination : float
        Inclination angle with respect to the plane of the sky
        (inclination = 0 corresponds to an edge-on bow shock).
    distance : float
        Source distance [pc]
    R0_corrected : float
        RS distance at the apex including thermal pressure [cm].

    Returns:
    --------
    dict
        Dictionary containing the stellar coordinates
    '''
    inc = np.deg2rad(inclination)

    x_R0 = -arcsecond(R0_corrected*np.cos(inc),distance)

    y_R0 = 0.0

    return {
        'x_R0': x_R0,
        'y_R0': y_R0
    }


def compute_plot_limits(extent, PA):
    '''
    Given an image (xmax-xmin)*(ymax-ymin)
    Calculates the new limits according to PA

    Parameters:
    -----------
    extent: list
    PA: float

    Returns:
    --------
    dict
    '''

    xmin, xmax, ymin, ymax = extent

    PA_rot = np.deg2rad(PA - 90.)

    corners = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax]
    ])

    x_rot = (
        corners[:,0]*np.cos(PA_rot)
        - corners[:,1]*np.sin(PA_rot)
    )

    y_rot = (
        corners[:,0]*np.sin(PA_rot)
        + corners[:,1]*np.cos(PA_rot)
    )

    return {
        'xmin': np.min(x_rot),
        'xmax': np.max(x_rot),
        'ymin': np.min(y_rot),
        'ymax': np.max(y_rot)
    }

def compute_map_extent(maps):
    """
    Compute the spatial extent of a the emission map.

    Parameters:
    -----------
    maps : dict
        Dictionary containing the map coordinate arrays
        ''maps['x']'' and ''maps['y']''.

    Returns:
    --------
    list
        Map extent in the format:
        [xmin, xmax, ymin, ymax].
    """

    return [
        np.min(maps['x']),
        np.max(maps['x']),
        np.min(maps['y']),
        np.max(maps['y'])
    ]


def update_map_image(ax, key, I_data, extent, images, colorbars, band_name, PA):
    """
    Create or update an emission map image.

    If the image does not exist, it is created together with its corresponding colorbar.
    If already exists, it updates the image data, including the normalization
    Takes into account PA rotation

    Parameters
    ----------
    ax :
        Axis where the map is displayed.
    key : str
        Identifier of the emission map (Halpha, OIII, or ff)
    I_data : ndarray
        Intensity map.
    extent : list
        Spatial extent of the image in the format [xmin, xmax, ymin, ymax].
    images : dict
        Dictionary storing image objects.
    colorbars : dict
        Dictionary storing colorbar objects.
    band_name : str
        Simultaed spectral band.
    PA : float
        PA angle
    """

    img = images[key]

    norm = get_norm(I_data)

    cmap = plt.colormaps['inferno'].copy()
    cmap.set_under('white')

    # PA rotation
    transform = (Affine2D().rotate_deg_around(0.0, 0.0, PA - 90.)+ ax.transData)

    if img is None:

        img_obj = ax.imshow(
                            I_data,
                            origin='lower',
                            extent=extent,
                            cmap=cmap,
                            norm=norm,
                            transform=transform
                        )

        images[key] = img_obj

        if key == 'ff' and (band_name == 'radio' or band_name == 'low_radio'):

            cbar_label = r'Surface brightness [mJy beam$^{-1}$]'

        elif key == 'Halpha':

            cbar_label = r'Surface brightness [R]'

        else:

            cbar_label = (
                r'Surface brightness '
                r'[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'
            )

        colorbars[key] = plt.colorbar(
            img_obj,
            ax=ax,
            label=cbar_label
        )

    else:

        img.set_data(I_data)
        img.set_extent(extent)
        img.set_norm(norm)
        img.set_transform(transform)


def update_star_marker(ax, x_star, y_star, key, star_markers):
    """
    Create or update the stellar position marker.

    Parameters
    ----------
    ax :
        Axis where the marker is displayed.
    x_star, y_star : float
        Stellar coordinates
    key : str
        Identifier of the emission map.
    star_markers : dict
        Dictionary storing star marker objects.
    """

    if star_markers[key] is None:

        star, = ax.plot(
            x_star,
            y_star,
            marker='*',
            color='black',
            markersize=10,
            zorder=5
        )

        star_markers[key] = star

    else:

        star_markers[key].set_data(
            [x_star],
            [y_star]
        )


def update_map_arrow(ax, key, x_star, y_star, x0, y0,
                    R0_corrected, distance, arrows, PA):
    """
    Create or update the arrow from the star to the apex.

    Parameters
    ----------
    ax : 
        Axis where the arrow is displayed.
    key : str
        Identifier of the emission map.
    x_star, y_star : float
        Stellar position
    x0, y0 : float
        Apex coordinates
    R0_corrected : float
        RS distance [cm]
    distance : float
        Source distance [pc]
    arrows : dict
        Dictionary storing arrow objects.
    PA : float
        Projected angle
    """

    if arrows[key] is not None:
        arrows[key].remove()

    # PA rotation
    transform = (
                Affine2D()
                .rotate_deg_around(0.0, 0.0, PA - 90.)
                + ax.transData
            )

    arr = ax.arrow(
                x_star,
                y_star,
                x0*1.5,
                y0*1.5,
                color='grey',
                width=0.3,
                head_width=0.15 * arcsecond(R0_corrected, distance),
                length_includes_head=True,
                zorder=5,
                transform=transform
            )

    arrows[key] = arr


def update_map_contours(ax, key, maps, I_data, contours, PA):
    """
    Create or update contour levels over an emission map.
    Contours are drawn at 0.05, 0.1, and 0.5 of the maximum intensity.

    Parameters
    ----------
    ax : 
        Axis where contours are displayed.
    key : str
        Identifier of the emission map.
    maps : dict
        Dictionary containing map coordinates.
    I_data : ndarray
        Intensity map used to compute contours.
    contours : dict
        Dictionary storing contours.
    PA : float
        Projected angle
    """

    if contours[key] is not None:
        contours[key].remove()

    # PA rotation
    transform = (
                Affine2D()
                .rotate_deg_around(0.0, 0.0, PA - 90.)
                + ax.transData
            )

    cont = ax.contour(
        maps['x'],
        maps['y'],
        I_data,
        levels=np.max(I_data)*np.array([0.1,0.25,0.5]),
        colors='lime',
        transform=transform
    )

    contours[key] = cont


def update_map_limits(map_axes, xmin, xmax, ymin, ymax):
    """
    Update axis limits and aspect ratio for all map panels.

    Parameters
    ----------
    map_axes : iterable
    xmin, xmax : float
        X-axis limits [arcsec].
    ymin, ymax : float
        Y-axis limits [arcsec].
    """

    for ax in map_axes:

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')


def update_map_panel(ax, key, I_data, extent, maps,
                    x_star, y_star, x0, y0,
                    R0_corrected, distance, band_name, PA,
                    images, colorbars, contours,
                    star_markers, arrows):
    """
    Update all visual elements of the emission maps:
    image, contours, stellar marker, and direction arrow.

    Parameters
    ----------
    ax
    key : str
        Identifier of the emission map.
    I_data : ndarray
        Intensity map to display.
    extent : list
        Spatial extent of the image [xmin, xmax, ymin, ymax].
    maps : dict
        Dictionary containing map coordinates.
    x_star, y_star : float
        Stellar coordinates [arcsec].
    x0, y0 : float
        Apex coordinates [arcsec].
    R0_corrected : float
        RS distance at the apex [cm].
    distance : float
        Source distance [pc].
    band_name : str
        Observing band for free-free emission.
    PA : float
        Projected angle (0°-> north, 90°-> east, counterclockwise)
    images : dict
        Dictionary storing image objects.
    colorbars : dict
        Dictionary storing colorbar objects.
    contours : dict
        Dictionary storing contour objects.
    star_markers : dict
        Dictionary storing stellar marker objects.
    arrows : dict
        Dictionary storing arrow objects.
    """

    update_map_image(ax, key, I_data, extent, images, colorbars, band_name, PA)

    update_map_contours(ax, key, maps, I_data, contours, PA)

    update_star_marker(ax, x_star, y_star, key, star_markers)

    update_map_arrow(ax, key, x_star, y_star, x0, y0, R0_corrected, distance, arrows, PA)