"""
BBC Mundo RSS feed parser — filters stories from the last 24 hours.

BBC Mundo offers these RSS feeds:
  - Portada:        https://www.bbc.co.uk/mundo/index.xml
  - Últimas:        https://www.bbc.co.uk/mundo/ultimas_noticias/index.xml
  - Internacional:  https://www.bbc.co.uk/mundo/temas/internacional/index.xml
  - América Latina:  https://www.bbc.co.uk/mundo/temas/america_latina/index.xml
"""

import logging
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

import requests

logger = logging.getLogger(__name__)

FEEDS = [
    # BBC Mundo
    "https://www.bbc.co.uk/mundo/index.xml",
    "https://www.bbc.co.uk/mundo/ultimas_noticias/index.xml",
    "https://www.bbc.co.uk/mundo/temas/internacional/index.xml",
    # El Mundo (Spain, international coverage)
    # "https://www.elmundo.es/rss/portada.xml",  # TODO: scraper doesn't work for elmundo — see issue #32
]


def parse_rss_datetime(date_str: str) -> datetime | None:
    """Parse RFC 822 / RFC 2822 date strings found in RSS <pubDate>."""
    if not date_str:
        return None
    # RSS dates are RFC 822 / RFC 2822 — parsedate_to_datetime handles them directly
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(date_str.strip())
    except Exception:
        return None


def fetch_stories(max_age_hours: int = 24) -> list[dict]:
    """
    Fetch all RSS feeds and return stories published within max_age_hours.
    Each dict: {title, link, description, pub_date, source}
    """
    cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
    cutoff_timestamp = cutoff.timestamp()
    all_stories = []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; bbc-noticias-bot/1.0)"}

    for feed_url in FEEDS:
        try:
            resp = requests.get(feed_url, headers=headers, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            # RSS 2.0 namespace
            channel = root.find("channel")
            if channel is None:
                continue

            source = channel.findtext("title", feed_url)

            for item in channel.findall("item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                description = item.findtext("description", "").strip()
                pub_date_str = item.findtext("pubDate", "")

                if not title or not link:
                    continue

                pub_date = parse_rss_datetime(pub_date_str)
                if pub_date is None:
                    logger.warning("Pub date could not be parsed: %s", pub_date_str)
                    continue

                # Filter by age
                if pub_date.timestamp() < cutoff_timestamp:
                    continue

                all_stories.append(
                    {
                        "title": title,
                        "link": link,
                        "description": description,
                        "pub_date": pub_date.isoformat(),
                        "source": source,
                    }
                )

        except Exception as e:
            logger.warning("[rss] Failed to fetch %s: %s", feed_url, e)

    return all_stories


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stories = fetch_stories()
    print(f"Found {len(stories)} stories from the last 24h:")
    for s in stories:
        print(f"  [{s['pub_date']}] {s['title']}")
        print(f"    {s['link']}")
