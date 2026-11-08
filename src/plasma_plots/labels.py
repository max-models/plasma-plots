"""Numbers in plot labels and titles, written the way a reader writes them.

>>> format_number(0.57825)
'0.578'
>>> format_number(0.0)
'0'
>>> format_number(1.234e-5)
'1.23×10⁻⁵'
"""

from __future__ import annotations

import math

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _strip(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_number(value: float, digits: int = 3) -> str:
    """Write a number for a label: plain between 0.001 and 10000, else as a power of ten.

    Parameters
    ----------
    value : float
        The number.
    digits : int, optional
        Significant digits. Default: 3.

    Returns
    -------
    str
        E.g. ``"0.578"``, ``"12.3"``, ``"1234"``, ``"0"``, ``"1.23×10⁻⁵"``; trailing zeros are
        dropped.

    Examples
    --------
    >>> format_number(5.7825e-01)
    '0.578'
    >>> format_number(2.0e6)
    '2×10⁶'
    """
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if -3 <= exponent < 4:
        decimals = max(digits - 1 - exponent, 0)
        return _strip(f"{value:.{decimals}f}")
    mantissa = _strip(f"{value / 10**exponent:.{digits - 1}f}")
    if mantissa in ("10", "-10"):  # rounding up to the next power
        mantissa, exponent = mantissa[:-1], exponent + 1
    power = f"10{str(exponent).translate(_SUPERSCRIPT)}"
    return (
        power
        if mantissa == "1"
        else f"-{power}"
        if mantissa == "-1"
        else f"{mantissa}×{power}"
    )
