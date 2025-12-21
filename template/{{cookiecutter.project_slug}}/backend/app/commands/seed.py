{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite -%}
# ruff: noqa: I001 - Imports structured for Jinja2 template conditionals
"""
Seed database with sample data.

This command is useful for development and testing.
Uses random data generation - install faker for better data:
    uv add faker --group dev
"""

import asyncio
import random
import string

import click
from sqlalchemy import delete, select

from app.commands import command, info, success, warning

# Try to import Faker for better data generation
try:
    from faker import Faker
    fake = Faker()
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False
    fake = None


def random_email() -> str:
    """Generate a random email address."""
    if HAS_FAKER:
        return fake.email()
    random_str = ''.join(random.choices(string.ascii_lowercase, k=8))
    return f"{random_str}@example.com"


def random_name() -> str:
    """Generate a random full name."""
    if HAS_FAKER:
        return fake.name()
    first_names = ["John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve", "Frank"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def random_title() -> str:
    """Generate a random item title."""
    if HAS_FAKER:
        return fake.sentence(nb_words=4).rstrip('.')
    adjectives = ["Amazing", "Great", "Awesome", "Fantastic", "Incredible", "Beautiful"]
    nouns = ["Widget", "Gadget", "Thing", "Product", "Item", "Object"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


def random_description() -> str:
    """Generate a random description."""
    if HAS_FAKER:
        return fake.paragraph(nb_sentences=3)
    return "This is a sample description for development purposes."


@command("seed", help="Seed database with sample data")
@click.option("--count", "-c", default=10, type=int, help="Number of records to create")
@click.option("--clear", is_flag=True, help="Clear existing data before seeding")
@click.option("--dry-run", is_flag=True, help="Show what would be created without making changes")
{%- if cookiecutter.use_jwt %}
@click.option("--users/--no-users", default=True, help="Seed users (default: True)")
{%- endif %}
{%- if cookiecutter.include_example_crud %}
@click.option("--items/--no-items", default=True, help="Seed items (default: True)")
{%- endif %}
def seed(
    count: int,
    clear: bool,
    dry_run: bool,
{%- if cookiecutter.use_jwt %}
    users: bool,
{%- endif %}
{%- if cookiecutter.include_example_crud %}
    items: bool,
{%- endif %}
) -> None:
    """
    Seed the database with sample data for development.

    Example:
        project cmd seed --count 50
        project cmd seed --clear --count 100
        project cmd seed --dry-run
    {%- if cookiecutter.use_jwt %}
        project cmd seed --no-users --items  # Only seed items
    {%- endif %}
    """
    if not HAS_FAKER:
        warning("Faker not installed. Using basic random data. For better data: uv add faker --group dev")

    if dry_run:
        info(f"[DRY RUN] Would create {count} sample records per entity")
        if clear:
            info("[DRY RUN] Would clear existing data first")
{%- if cookiecutter.use_jwt %}
        if users:
            info("[DRY RUN] Would create users")
{%- endif %}
{%- if cookiecutter.include_example_crud %}
        if items:
            info("[DRY RUN] Would create items")
{%- endif %}
        return

{%- if cookiecutter.use_postgresql %}
    from app.db.session import async_session_maker
{%- if cookiecutter.use_jwt %}
    from app.db.models.user import User
    from app.core.security import get_password_hash
{%- endif %}
{%- if cookiecutter.include_example_crud %}
    from app.db.models.item import Item
{%- endif %}

    async def _seed():
        async with async_session_maker() as session:
            created_counts = {}

{%- if cookiecutter.use_jwt %}
            # Seed users
            if users:
                if clear:
                    info("Clearing existing users (except superusers)...")
                    await session.execute(delete(User).where(User.is_superuser == False))  # noqa: E712
                    await session.commit()

                # Check how many users already exist
                result = await session.execute(select(User).limit(1))
                existing = result.scalars().first()

                if existing and not clear:
                    info("Users already exist. Use --clear to replace them.")
                else:
                    info(f"Creating {count} sample users...")
                    for _ in range(count):
                        user = User(
                            email=random_email(),
                            hashed_password=get_password_hash("password123"),
                            full_name=random_name(),
                            is_active=True,
                            is_superuser=False,
                            role="user",
                        )
                        session.add(user)
                    await session.commit()
                    created_counts["users"] = count
{%- endif %}

{%- if cookiecutter.include_example_crud %}
            # Seed items
            if items:
                if clear:
                    info("Clearing existing items...")
                    await session.execute(delete(Item))
                    await session.commit()

                # Check how many items already exist
                result = await session.execute(select(Item).limit(1))
                existing = result.scalars().first()

                if existing and not clear:
                    info("Items already exist. Use --clear to replace them.")
                else:
                    info(f"Creating {count} sample items...")
                    for _ in range(count):
                        item = Item(
                            title=random_title(),
                            description=random_description(),
                            is_active=random.choice([True, True, True, False]),  # 75% active
                        )
                        session.add(item)
                    await session.commit()
                    created_counts["items"] = count
{%- endif %}

            if created_counts:
                summary = ", ".join(f"{v} {k}" for k, v in created_counts.items())
                success(f"Created: {summary}")
            else:
                info("No records created.")

    asyncio.run(_seed())
{%- elif cookiecutter.use_sqlite %}
    from app.db.session import SessionLocal
{%- if cookiecutter.use_jwt %}
    from app.db.models.user import User
    from app.core.security import get_password_hash
{%- endif %}
{%- if cookiecutter.include_example_crud %}
    from app.db.models.item import Item
{%- endif %}

    with SessionLocal() as session:
        created_counts = {}

{%- if cookiecutter.use_jwt %}
        # Seed users
        if users:
            if clear:
                info("Clearing existing users (except superusers)...")
                session.execute(delete(User).where(User.is_superuser == False))  # noqa: E712
                session.commit()

            # Check how many users already exist
            result = session.execute(select(User).limit(1))
            existing = result.scalars().first()

            if existing and not clear:
                info("Users already exist. Use --clear to replace them.")
            else:
                info(f"Creating {count} sample users...")
                for _ in range(count):
                    user = User(
                        email=random_email(),
                        hashed_password=get_password_hash("password123"),
                        full_name=random_name(),
                        is_active=True,
                        is_superuser=False,
                        role="user",
                    )
                    session.add(user)
                session.commit()
                created_counts["users"] = count
{%- endif %}

{%- if cookiecutter.include_example_crud %}
        # Seed items
        if items:
            if clear:
                info("Clearing existing items...")
                session.execute(delete(Item))
                session.commit()

            # Check how many items already exist
            result = session.execute(select(Item).limit(1))
            existing = result.scalars().first()

            if existing and not clear:
                info("Items already exist. Use --clear to replace them.")
            else:
                info(f"Creating {count} sample items...")
                for _ in range(count):
                    item = Item(
                        title=random_title(),
                        description=random_description(),
                        is_active=random.choice([True, True, True, False]),  # 75% active
                    )
                    session.add(item)
                session.commit()
                created_counts["items"] = count
{%- endif %}

        if created_counts:
            summary = ", ".join(f"{v} {k}" for k, v in created_counts.items())
            success(f"Created: {summary}")
        else:
            info("No records created.")
{%- endif %}
{%- endif %}
