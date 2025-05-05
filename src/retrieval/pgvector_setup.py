from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import getpass

DB_URL = f"postgresql://{getpass.getuser()}@localhost/postgres"

engine = create_engine(DB_URL)
Base = declarative_base()

from pgvector.sqlalchemy import Vector
import psycopg2

class TextEmbedding(Base):
    __tablename__ = "text_embeddings"
    id = Column(Integer, primary_key=True)
    timestamp = Column(Float)
    embedding = Column(Vector(384))  # 384 = MiniLM-L6 output size

Base.metadata.create_all(engine)

# Load data
embeddings = np.load("data/embeddings/text/text_embeddings.npy")
timestamps = np.load("data/embeddings/text/timestamps.npy")

Session = sessionmaker(bind=engine)
session = Session()

for i in range(len(embeddings)):
    row = TextEmbedding(
        timestamp=float(timestamps[i]),
        embedding=embeddings[i].tolist()
    )
    session.add(row)

session.commit()
session.close()

print("✅ Embeddings loaded into PostgreSQL with pgvector.")