import asyncio
import os
import re

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = os.environ.get(
    "AMORTIZED_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized",
)

ASYNC_URL = re.sub(r"^postgresql(\+\w+)?://", "postgresql+asyncpg://", DATABASE_URL)

target_metadata = None


def run_migrations_offline() -> None:
    context.configure(url=DATABASE_URL, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = create_async_engine(ASYNC_URL, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(run_migrations_online())
    else:
        asyncio.run(run_migrations_online())
