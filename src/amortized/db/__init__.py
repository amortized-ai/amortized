"""Database layer for job and artifact persistence."""

from amortized.config import settings as settings
from amortized.db.connection import get_db as get_db
from amortized.db.connection import init_db as init_db
from amortized.db.repository import Repository as Repository
