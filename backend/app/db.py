from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .config import Settings


class MongoStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: AsyncIOMotorClient | None = None
        self.database: AsyncIOMotorDatabase | None = None

    async def connect(self) -> None:
        self.client = AsyncIOMotorClient(self.settings.mongodb_uri, serverSelectionTimeoutMS=5000)
        await self.client.admin.command("ping")
        self.database = self.client[self.settings.mongodb_database]
        await self._create_indexes()

    async def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
            self.database = None

    async def _create_indexes(self) -> None:
        assert self.database is not None
        await self.database.products.create_index("created_at")
        await self.database.campaigns.create_index([("product_id", 1), ("created_at", -1)])
        await self.database.tasks.create_index([("campaign_id", 1), ("status", 1)])
        await self.database.leads.create_index([("campaign_id", 1), ("created_at", -1)])
        await self.database.leads.create_index("email")
        await self.database.leads.create_index([("weekly_budget", 1), ("created_at", -1)])
        await self.database.leads.create_index([("stack", 1), ("created_at", -1)])
        await self.database.speed_demo_requests.create_index([("created_at", -1)])
        await self.database.speed_demo_requests.create_index("phone")
        await self.database.analytics.create_index([("campaign_id", 1), ("recorded_at", -1)])

    def collection(self, name: str):
        if self.database is None:
            raise RuntimeError("MongoDB is not connected")
        return self.database[name]


@asynccontextmanager
async def lifespan_store(settings: Settings) -> AsyncIterator[MongoStore]:
    store = MongoStore(settings)
    await store.connect()
    try:
        yield store
    finally:
        await store.close()
