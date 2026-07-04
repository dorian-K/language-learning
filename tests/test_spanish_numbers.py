"""Correctness tests for the deterministic Spanish number speller.

conftest.py already puts src/ on sys.path, so spanish_numbers imports directly.
"""

import pytest

from spanish_numbers import apocope, cardinal, feminine, format_numeral, ordinal


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        # building blocks 0-15 (unique words)
        (0, "cero"),
        (1, "uno"),
        (11, "once"),
        (15, "quince"),
        # 16-19 and 21-29 written together
        (16, "dieciséis"),
        (19, "diecinueve"),
        (20, "veinte"),
        (21, "veintiuno"),
        (22, "veintidós"),
        (29, "veintinueve"),
        # y-joining in 31-99
        (30, "treinta"),
        (31, "treinta y uno"),
        (47, "cuarenta y siete"),
        (99, "noventa y nueve"),
        # cien vs ciento
        (100, "cien"),
        (101, "ciento uno"),
        (115, "ciento quince"),
        # irregular hundreds
        (200, "doscientos"),
        (500, "quinientos"),
        (700, "setecientos"),
        (900, "novecientos"),
        (256, "doscientos cincuenta y seis"),
        (999, "novecientos noventa y nueve"),
        # thousands / millions
        (1000, "mil"),
        (1234, "mil doscientos treinta y cuatro"),
        (2015, "dos mil quince"),
        (100000, "cien mil"),
        (999999, "novecientos noventa y nueve mil novecientos noventa y nueve"),
        (1000000, "un millón"),
        (2500000, "dos millones quinientos mil"),
        (1000000000, "mil millones"),
    ],
)
def test_cardinal(n, expected):
    assert cardinal(n) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        # uno apocopates to un/veintiún directly before mil/millón — the num2words bug we fix
        (21000, "veintiún mil"),
        (31000, "treinta y un mil"),
        (21000000, "veintiún millones"),
        (101000, "ciento un mil"),
        (121000, "ciento veintiún mil"),
        (21000000000, "veintiún mil millones"),
    ],
)
def test_cardinal_apocope_before_mil(n, expected):
    assert cardinal(n) == expected


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("uno", "una"),
        ("veintiuno", "veintiuna"),
        ("treinta y uno", "treinta y una"),
        ("doscientos", "doscientas"),
        ("quinientos", "quinientas"),
        ("novecientos", "novecientas"),
        ("doscientos mil", "doscientas mil"),
        # invariable forms untouched
        ("cien", "cien"),
        ("ciento uno", "ciento una"),
    ],
)
def test_feminine(spelling, expected):
    assert feminine(spelling) == expected


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("uno", "un"),
        ("veintiuno", "veintiún"),
        ("treinta y uno", "treinta y un"),
    ],
)
def test_apocope(spelling, expected):
    assert apocope(spelling) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "primero"),
        (2, "segundo"),
        (3, "tercero"),
        (5, "quinto"),
        (10, "décimo"),
    ],
)
def test_ordinal(n, expected):
    assert ordinal(n) == expected


def test_ordinal_apocope_and_gender():
    assert ordinal(1, apocopate=True) == "primer"
    assert ordinal(3, apocopate=True) == "tercer"
    assert ordinal(1, gender="f") == "primera"
    assert ordinal(3, gender="f") == "tercera"
    # apocope only applies to 1st/3rd; others fall back to full form
    assert ordinal(5, apocopate=True) == "quinto"


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (5, "5"),
        (1000, "1.000"),
        (21000, "21.000"),
        (1000000, "1.000.000"),
        (1000000000, "1.000.000.000"),
    ],
)
def test_format_numeral(n, expected):
    assert format_numeral(n) == expected
