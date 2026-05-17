"""Initialize database tables and baseline scaler state.

This script runs before the runtime services in Docker Compose.
"""
from datetime import datetime

from sqlalchemy import inspect, text

from shared.database import SessionLocal, engine
from shared.models import ActiveReplicaState, Base


Base.metadata.create_all(bind=engine)

inspector = inspect(engine)
if "forecast_points" in inspector.get_table_names():
    forecast_columns = {column["name"] for column in inspector.get_columns("forecast_points")}
    if "algorithm" not in forecast_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE forecast_points ADD COLUMN algorithm VARCHAR(32) DEFAULT 'unknown'"))

with SessionLocal() as session:
    state = session.get(ActiveReplicaState, 1)
    if state is None:
        state = ActiveReplicaState(id=1, active_replicas=1, mode="reactive", updated_at=datetime.utcnow())
        session.add(state)
        session.commit()

print("Database initialized.")
