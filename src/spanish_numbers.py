"""Deterministic Spanish number spelling.

Spanish numbers are a closed, algorithmic system, so we spell them in pure Python
instead of asking an LLM (LLMs misspell large numbers). This wraps ``num2words``
for the heavy lifting of composition and fixes the cases it gets wrong for Spanish:
the apocope of ``uno`` -> ``un`` / ``veintiuno`` -> ``veintiún`` directly before
``mil`` / ``millón`` / ``millones``.

Public helpers:
    cardinal(n)        -> masculine cardinal, e.g. 47 -> "cuarenta y siete"
    feminine(spelling) -> gender-agreeing form,  doscientos -> "doscientas"
    apocope(spelling)  -> pre-masc-noun form,     veintiuno -> "veintiún"
    ordinal(n, ...)    -> ordinal,                3 -> "tercero" / "tercer"
    format_numeral(n)  -> grouped digits,         1000000 -> "1.000.000"
"""

import re

from num2words import num2words

_MIL_MILLON = r"(mil|millón|millones)"


def cardinal(n: int) -> str:
    """Masculine cardinal spelling of ``n`` (num2words + Spanish apocope fixes)."""
    words = num2words(n, lang="es")
    # num2words leaves "veintiuno mil", "treinta y uno mil", "veintiuno millones"
    # un-apocopated. Correct Spanish apocopates "uno" before mil/millón.
    words = re.sub(rf"veintiuno (?={_MIL_MILLON})", "veintiún ", words)
    words = re.sub(rf"\buno (?={_MIL_MILLON})", "un ", words)
    return words


def feminine(spelling: str) -> str:
    """Feminine-agreeing form of a cardinal spelling.

    ``uno`` -> ``una``, ``veintiuno`` -> ``veintiuna`` (also "... y uno" -> "... y una"),
    and the hundreds ``-cientos`` -> ``-cientas`` (doscientos -> doscientas ...).
    ``cien`` / ``ciento`` / ``millón`` are invariable and left untouched.
    """
    spelling = re.sub(r"veintiuno\b", "veintiuna", spelling)
    spelling = re.sub(r"\buno\b", "una", spelling)
    # all hundreds 200-900 end in "-ientos" (doscientos ... quinientos ... novecientos)
    spelling = re.sub(r"ientos\b", "ientas", spelling)
    return spelling


def apocope(spelling: str) -> str:
    """Apocopated form used directly before a masculine noun.

    ``uno`` -> ``un``, ``veintiuno`` -> ``veintiún`` (also "... y uno" -> "... y un").
    """
    spelling = re.sub(r"veintiuno\b", "veintiún", spelling)
    spelling = re.sub(r"\buno\b", "un", spelling)
    return spelling


# Ordinals 1-10 have irregular apocope (primer/tercer) used before a masculine noun.
_ORDINAL_APOCOPE = {"primero": "primer", "tercero": "tercer"}


def ordinal(n: int, gender: str = "m", apocopate: bool = False) -> str:
    """Ordinal spelling of ``n`` (primero, segundo, tercero ...).

    ``gender="f"`` yields the feminine form (primera, segunda ...).
    ``apocopate=True`` yields primer/tercer for 1st/3rd (masculine only).
    """
    word = num2words(n, lang="es", to="ordinal")
    if apocopate and word in _ORDINAL_APOCOPE:
        return _ORDINAL_APOCOPE[word]
    if gender == "f":
        return re.sub(r"o\b", "a", word)
    return word


def format_numeral(n: int) -> str:
    """Format an integer with Spanish ``.`` thousands separators (1000000 -> 1.000.000)."""
    return f"{n:,}".replace(",", ".")
