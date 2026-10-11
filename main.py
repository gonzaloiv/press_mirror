"""Production WSGI entry point.

Render start command: ``gunicorn main:app``.
"""

from web_app.server import app

__all__ = ["app"]
