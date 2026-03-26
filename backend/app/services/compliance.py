from urllib.parse import urlparse

from app.config import settings


def is_url_allowlisted(url: str) -> tuple[bool, str]:
    host = urlparse(url).netloc.lower()
    allowed = any(host == domain or host.endswith(f".{domain}") for domain in settings.allowed_domains)
    return allowed, host
