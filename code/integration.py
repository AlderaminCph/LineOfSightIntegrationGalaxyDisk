"""
Module for integration 3D luminosity density model of exponential disk and for
creating analytic 2D edge-on disk model using van der Cruit formulae.
"""
# !/usr/bin/env python
# coding: utf-8

import math
import numbers
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.special import k1


@dataclass
class Point:
    """
    2 dimensional point
    """

    x: float
    y: float


@dataclass
class Bounds:
    """
    Limits of integration.
    """

    lower: float
    upper: float


@dataclass
class GalaxyProperties:
    """
    Parameters of galaxy.
    """

    pos_angle: float
    central_lum_density: float
    h_scale: float
    z_0_scale: float
    n_value: float


def gauss_legendre(
        luminosity_density,
        limits: Optional[Bounds],
        parameters: List[float],
        roots: List[float],
        weights: List[float],
) -> float:
    """Integrate luminosity_density within integration bounds using gauss legendre quadrature.
    Args:
        parameters: the list of parameters for integrand function
        luminosity_density: integrand function
        limits: a Bounds object with lower and upper integration limit
        roots: roots of the gauss legendre polynom
        weights: weights of the gauss legendre polynom
    Return:
        the integral value
    """
    new_root = ((limits.upper - limits.lower) / 2) * roots + (
        (limits.upper + limits.lower) / 2
    )
    return ((limits.upper - limits.lower) / 2) * sum(
        weights * luminosity_density(new_root, parameters)
    )


def make_model(
        model_type: str,
        frame_size: int,
        parameters: List[float],
        roots=None,
        weights=None,
        quad=False,
) -> List[List]:
    """Return the 2D array with total intensity values and size of the framesize.
    Args:
        model_type: name of the model (exp_disk_3D or edge_on)
        frame_size: the size of the grid
        parameters: list of function parameters p = [x0, y0, PA, inc, J_0, h, n, z_0]
            centre_pixel: Point object with x and y pixel coordinate of the centre
            PA: major-axis position angle (in degrees)
            inc: inclination with respect to the line of sight (in degrees)
            J_0: the central luminosity density
            h: exponential scale length
            n: the vertical profile follows the sech^(2/n) function
            z_0: the vertical scale height
            quad: a boolean flag. If it's True the scipy.inegrate.quad is used
    Return:
        TotalIntensity: the array with the total intensity value.
    """
    centre_pixel = Point(frame_size / 2, frame_size / 2)
    total_intensity = []
    parameters.insert(0, centre_pixel)
    for x_coord in np.arange(frame_size):
        intensity = []
        for y_coord in np.arange(frame_size):
            if model_type == "exp_disk_3D":
                intensity.append(
                    exponential_disk_3d(
                        Point(x_coord, y_coord), parameters, roots, weights, quad
                    )
                )
            elif model_type == "edge_on":
                intensity.append(edge_on_disk(Point(x_coord, y_coord), parameters))
        total_intensity.append(intensity)
    return total_intensity


def exponential_disk_3d(
        pixel: Optional[Point],
        parameters: List[float],
        roots: List[float],
        weights: List[float],
        quad=False,
) -> float:
    """Get intensity of the pixel with coordinates x, y in model
        of 3D exponential disk.
    Args:
        pixel: Point object with x and y coordinate of pixel
        parameters: list of function parameters p = [centre_pixel, pos_angle, inc, J_0, h, n, z_0]
            centre_pixel: Point object with x and y pixel coordinate of the centre
            pos_angle: major-axis position angle (in degrees)
            inc: inclination with respect to the line of sight (in degrees)
            J_0: the central luminosity density
            h_scale: exponential scale length
            n: the vertical profile follows the sech^(2/n) function
            z_0: the vertical scale height
        roots: roots of the gauss legendre polynom
        weights: weights of the gauss legendre polynom
        quad: a boolean flag. If it's True the scipy.integrate.quad is used
    Return:
        intensity: intensity of the pixel
    """
    centre_pixel, pos_angle, inc, central_lum_density, h_scale, n, z_0 = parameters
    pos_angle += 90
    inc_rad = math.radians(inc)
    cos_inc = math.cos(inc_rad)
    sin_inc = math.sin(inc_rad)

    projected_pix = get_reference_frame_coords(pixel, centre_pixel, pos_angle)

    # Calculate (x,y,z)_start in component's native xyz reference frame,
    # corresponding to intersection of line-of-sight ray with projected
    # sky frame
    x_d0 = projected_pix.x
    y_d0 = projected_pix.y * cos_inc
    z_d0 = projected_pix.y * sin_inc

    param = [x_d0, y_d0, z_d0, cos_inc, sin_inc, central_lum_density, h_scale, z_0, n]
    if inc < 80:
        # intersection of the disk plane and sky plane
        s_plane = projected_pix.y * np.tan(inc_rad)
    else:
        s_plane = 0
    # integrate interval
    interval = max(h_scale * np.sin(inc_rad), z_0 * np.cos(inc_rad))

    inner, outer = interval, interval
    inner *= 5
    outer *= 20

    if not quad:
        intensity = (
            gauss_legendre(
                get_luminosity_density,
                Bounds(s_plane - outer, s_plane - inner),
                param,
                roots,
                weights,
            )
            + gauss_legendre(
                get_luminosity_density,
                Bounds(s_plane - inner, s_plane),
                param,
                roots,
                weights,
            )
            + gauss_legendre(
                get_luminosity_density,
                Bounds(s_plane, s_plane + inner),
                param,
                roots,
                weights,
            )
            + gauss_legendre(
                get_luminosity_density,
                Bounds(s_plane + inner, s_plane + outer),
                param,
                roots,
                weights,
            )
        )
    else:
        intensity = integrate.quad(get_luminosity_density, -100, 100, args=param)[0]
    return intensity


def get_luminosity_density(
        line_of_sight_coord: float, parameters: List[float]
) -> float:
    """Get luminosity density along line of sight coordinate s.
    Args:
        line_of_sight_coord (float): line of sight coordinate
        parameters: list of function parameters
            parameters = [x_d0, y_d0, z_d0, cosInc, sinInc, J_0, h, z_0]
            x_d0: x coordinate in native xyz reference frame
            y_d0: y coordinate in native xyz reference frame
            z_d0: z coordinate in native xyz reference frame
            cosInc: cos of inclination with respect to the LoS (in degrees)
            sinInc: sin of inclination with respect to the LoS (in degrees)
            J_0: the central luminosity density
            h: exponential scale length
            z_0: the vertical scale height
    Return:
        lum_density: luminosity density along line of sight coordinate line_of_sight_coord.
    """

    (
        x_d0,
        y_d0,
        z_d0,
        cos_inc,
        sin_inc,
        central_lum_density,
        h_scale,
        z_0,
        n,
    ) = parameters

    x_d = x_d0
    y_d = y_d0 + line_of_sight_coord * sin_inc
    z_d = z_d0-line_of_sight_coord * cos_inc

    r_coord = np.sqrt(x_d**2 + y_d**2)
    z_coord = abs(z_d)

    vertical_scaling = get_vertical_scaling(n, z_0, z_coord)

    lum_density = central_lum_density * np.exp(-r_coord / h_scale) * vertical_scaling
    return lum_density


def numeric_solution(roots, weights):
    """
    Calculate numeric solution for integration of 3D exponential disk
    Args:
        roots: roots of the gauss legendre polynom
        weights: weights of the gauss legendre polynom

    Returns:
        total_intensity: intensity array of the model
    """
    frame_size = 100  # the size of the grid

    parameters = setup_galaxy("exp_disk_3D")

    total_intensity = make_model("exp_disk_3D", frame_size, parameters, roots, weights)
    plot_intensity_map(total_intensity, "Numeric", "Numeric.png")

    return total_intensity


def get_reference_frame_coords(
        pixel: Optional[Point], centre_pixel: Optional[Point], pos_angle: float
) -> Optional[Point]:
    """
    Calculate coordinates in component reference frame
    Args:
        pixel: Point object with current x and y pixel coordinate
        centre_pixel: Point object with x and y pixel coordinate of the centre
        pos_angle: major-axis position angle (in degrees)

    Returns:
        projected_point: x and y coordinate in the reference frame
    """
    x_diff = pixel.x - centre_pixel.x
    y_diff = pixel.y - centre_pixel.y
    pa_rad = math.radians(pos_angle)
    cos_pa = math.cos(pa_rad)
    sin_pa = math.sin(pa_rad)
    x_projected = x_diff * cos_pa + y_diff * sin_pa
    y_projected = -x_diff * sin_pa + y_diff * cos_pa
    projected_point = Point(x_projected, y_projected)
    return projected_point


def get_vertical_scaling(n: float, z_0: float, z_coord: float):
    """
    Calculate the multiplier
    Args:
        n: the vertical profile follows the sech^(2/n) function
        z_0: the vertical scale height
        z_coord: z coordinate

    Returns:
        vertical_scaling: a multiplier
    """
    cosh_limit = 100
    alpha = 2.0 / n
    scaled_z0 = alpha * z_0
    two_to_alpha = pow(2.0, alpha)
    condition = z_coord / scaled_z0
    if not isinstance(condition, numbers.Number):
        condition = condition.any()
    if condition > cosh_limit:
        vertical_scaling = two_to_alpha * np.exp(-z_coord / z_0)
    else:
        sech = 1.0 / np.cosh(z_coord / scaled_z0)
        vertical_scaling = pow(sech, alpha)
    return vertical_scaling


def edge_on_disk(pixel: Optional[Point], parameters: List[float]) -> float:
    """Provides the analytic form for an edge-on
    disk with a radial exponential profile, using the Bessel-function
    solution of van der Kruit & Searle (1981) for the radial profile.

        I(r,z) = mu(0,0) * (r/h) * K_1(r/h) * sech^(2/n)(n*z/(2*z0))
    where
        mu(0,0) = 2 * h * J_0
        K_1(r/h) - Modified Bessel function of the second kind of order 1
    Args:
        pixel: Point object with x and y coordinate of pixel
        parameters: list of function parameters p = [x0, y0, PA, inc, J_0, h, n, z_0]
            x0: x pixel coordinate of the centre
            y0: y pixel coordinate of the centre
            PA: major-axis position angle (in degrees)
            J_0: the central luminosity density
            h: exponential scale length
            n: the vertical profile follows the sech^(2/n) function
            z_0: the vertical scale height
    Return:
        intensity: intensity of the pixel with coordinates x,y
    """
    pixel_center, pos_ang, central_lum_density, h_scale, n, z_0 = parameters

    projected_point = get_reference_frame_coords(pixel, pixel_center, pos_ang + 90)
    r_coord, z_coord = np.fabs(projected_point.x), np.fabs(projected_point.y)
    # Calculate r,z in coordinate system aligned with the edge-on disk

    mu_0 = 2 * h_scale * central_lum_density  # the central surface brightness

    vertical_scaling = get_vertical_scaling(n, z_0, z_coord)

    if r_coord == 0:
        intensity = mu_0
    else:
        intensity = (
            mu_0 * (r_coord / h_scale) * k1(r_coord / h_scale) * vertical_scaling
        )

    return intensity


def setup_galaxy(model_type: str, inc=90) -> List[float]:
    """Setup galaxy parameters for 3D exponential disk and 2D Edge On disk
    models.
    Return:
        parameters: the vector of model's parameters.
    """
    gal_prop = GalaxyProperties(
        pos_angle=0, central_lum_density=0.01, h_scale=30, n_value=1, z_0_scale=5
    )
    if model_type == "exp_disk_3D":
        parameters = [
            gal_prop.pos_angle,
            inc,
            gal_prop.central_lum_density,
            gal_prop.h_scale,
            gal_prop.n_value,
            gal_prop.z_0_scale,
        ]
    elif model_type == "edge_on":
        parameters = [
            gal_prop.pos_angle,
            gal_prop.central_lum_density,
            gal_prop.h_scale,
            gal_prop.n_value,
            gal_prop.z_0_scale,
        ]
    return parameters


def plot_intensity_map(total_intensity, title, figname):
    """Plot intensity map"""
    plt.figure()
    plt.imshow(
        total_intensity,
        origin="lower",
        interpolation="nearest",
        vmin=np.min(total_intensity),
        vmax=np.max(total_intensity),
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Brightness", rotation=270, labelpad=25)

    plt.savefig(figname)
    plt.show()


def plot_edge_on():
    """plot edge on model."""
    frame_size = 100  # the size of the grid

    parameters = setup_galaxy("edge_on")

    total_intensity = np.array(make_model("edge_on", frame_size, parameters))

    plot_intensity_map(
        total_intensity,
        "Bessel-function solution EDGE-ON",
        "EDGE_ON_ANALYTIC_SOLUTION.png",
    )
    return total_intensity


ROOTS, WEIGHTS = np.polynomial.legendre.leggauss(70)
NUMERIC_INTENSITY = numeric_solution(ROOTS, WEIGHTS)
ANALYTIC_INTENSITY = plot_edge_on()
ax = []
fig = plt.figure(figsize=(12, 4))

ax.append(fig.add_subplot(1, 3, 1))
plt.imshow(ANALYTIC_INTENSITY, origin='lower')
ax[0].set_title("analytic")

ax.append(fig.add_subplot(1, 3, 2))
ax[1].set_title("numeric")
plt.imshow(NUMERIC_INTENSITY, origin='lower')

ax.append(fig.add_subplot(1, 3, 3))
ax[2].set_title("analytic - numeric")
plt.imshow(ANALYTIC_INTENSITY-NUMERIC_INTENSITY, origin='lower')
cbar = plt.colorbar()
cbar.set_label("Brightness", rotation=270, labelpad=25)

plt.savefig("python.png")
plt.show()
