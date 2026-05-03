"""Initialize database tables and baseline scaler state."""
from datetime import datetime

from shared.database import SessionLocal, engine
from shared.models import ActiveReplicaState, Base


Base.metadata.create_all(bind=engine)

with SessionLocal() as session:
    state = session.get(ActiveReplicaState, 1)
    if state is None:
        state = ActiveReplicaState(id=1, active_replicas=1, mode="reactive", updated_at=datetime.utcnow())
        session.add(state)
        session.commit()

print("Database initialized.")
