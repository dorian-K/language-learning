"""
Article scraper — fetches a BBC Mundo article URL and extracts readable text.
Uses basic regex + HTML parsing to strip navigation/clutter.
"""

import logging
import re

import requests

# BBC Mundo uses this User-Agent to get the article page
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9",
}


def fetch_article(url: str, timeout: int = 15) -> str | None:
    """
    Fetch a BBC Mundo article and return the main body text as plain Spanish.
    Returns None on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        # Extract the <article> body via a simple approach:
        # BBC uses <article data-component="article-body"> ... </article>
        body = _extract_article_body(html)
        if not body:
            # Fallback: grab everything between two common article markers
            body = _fallback_extract(html)

        if not body:
            return None

        # Clean up: strip tags, normalize whitespace
        text = _clean_html(body)
        return text if text else None

    except Exception as e:
        logger.warning("[scraper] Failed to fetch %s: %s", url, e)
        return None


def _extract_article_body(html: str) -> str | None:
    """Try to extract the main article body via data-component or class markers."""
    # Pattern 1: <article ...> ... </article>
    m = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)

    # Pattern 2: data-component="article-body"
    m = re.search(
        r'<div[^>]*data-component="article-body"[^>]*>(.*?)</div>', html, re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1)

    # Pattern 3: section with id="body-content"
    m = re.search(
        r'<section[^>]*id="body-content"[^>]*>(.*?)</section>', html, re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1)

    return None


def _fallback_extract(html: str) -> str | None:
    """Last-resort extraction from main content area."""
    m = re.search(r"<main[^>]*>(.*?)</main>", html, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _clean_html(html: str) -> str:
    """Remove HTML tags and normalize whitespace from article body."""
    # Remove script, style, nav, footer, aside, form
    html = re.sub(
        r"<(script|style|nav|footer|aside|form)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove sidebar/chrome sections: "Más leídas", "Podcast", related links
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)  # HTML comments
    html = re.sub(r"<aside[^>]*>.*?</aside>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(
        r'<div[^>]*class="[^"]*(?:sidebar|related|mas-leidas|más-leídas|recommended)[^"]*"[^>]*>.*?</div>',
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove "Saltar X y continuar leyendo" nav markers
    html = re.sub(r"Saltar\s+\w+[\s\w]*y continuar leyendo", "", html, flags=re.IGNORECASE)

    # Remove video / audio player placeholders
    html = re.sub(r"Para ver este contenido[^\n<]*", "", html, flags=re.IGNORECASE)

    # Replace block elements with newlines
    html = re.sub(r"<(p|div|br|h[1-6]|li|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    html = html.replace("\u00a0", " ")
    for entity, char in [
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&ntilde;", "ñ"),
        ("&Ntilde;", "Ñ"),
        ("&uuml;", "ü"),
        ("&aacute;", "á"),
        ("&eacute;", "é"),
        ("&iacute;", "í"),
        ("&oacute;", "ó"),
        ("&uacute;", "ú"),
        ("&iexcl;", "¡"),
        ("&iquest;", "¿"),
    ]:
        html = html.replace(entity, char)
    # Normalize whitespace: collapse multiple spaces, strip line ends
    html = re.sub(r" {2,}", " ", html)

    # Post-clean: remove common noise lines
    lines = [ln.strip() for ln in html.splitlines()]
    # Filter out BBC UI noise, photo credits, newsletter CTAs, sidebar blocks, empty lines
    skip_phrases = (
        "Fuente de la imagen",
        "Pie de foto",
        "Suscríbete aquí",
        "También puedes seguirnos",
        "Y recuerda que puedes recibir",
        "Haz clic aquí",
        "Más leídas",
        "Final de Más leídas",
        "Información del artículo",
        "Episodios",
        "Para ver este contenido",
        "Play video",
        "Título del video",
        "Duración",
        "Tiempo de lectura",
        "Actualizado",
    )
    skip_starts = (
        "Autor,",
        "Título del autor,",
        "https://www.bbc.com/mundo/",
    )
    filtered = []
    in_mas_leidas = False
    in_podcast = False
    for ln in lines:
        txt = ln.strip()
        # Skip "Más leídas" block (starts with headline, ends with "Final de Más leídas")
        if "Más leídas" in txt and "Final de Más leídas" not in txt:
            in_mas_leidas = True
            continue
        if in_mas_leidas:
            if "Final de Más leídas" in txt:
                in_mas_leidas = False
            continue
        # Skip podcast block
        if "El nuevo podcast de BBC Mundo" in txt:
            in_podcast = True
            continue
        if in_podcast:
            if "Fin de Podcast" in txt:
                in_podcast = False
            continue
        # Skip by phrase
        if any(p in txt for p in skip_phrases):
            continue
        # Skip by prefix
        if any(txt.startswith(p) for p in skip_starts):
            continue
        # Skip standalone date/readtime lines
        if re.match(r"^\d{1,2}\s+\w+\s+\d{4}$", txt):
            continue
        # Skip standalone video/audio duration lines like "02:27"
        if re.match(r"^\d{1,2}:\d{2}$", txt):
            continue
        if txt:
            filtered.append(ln)

    return "\n".join(filtered)


if __name__ == "__main__":
    # Quick smoke test
    url = "https://www.bbc.com/mundo/articles/c4gl71dez8zo"
    text = fetch_article(url)
    if text:
        logger.info("[scraper] Smoke test succeeded for %s", url)
    else:
        logger.warning("[scraper] Smoke test failed for %s", url)
