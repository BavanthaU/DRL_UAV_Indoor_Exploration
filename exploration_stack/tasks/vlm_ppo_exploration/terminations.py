from __future__ import annotations


def coverage_success(coverage_ratio, threshold: float):
    return coverage_ratio >= threshold


def altitude_out_of_bounds(altitude, min_altitude: float, max_altitude: float):
    return (altitude < min_altitude) | (altitude > max_altitude)

