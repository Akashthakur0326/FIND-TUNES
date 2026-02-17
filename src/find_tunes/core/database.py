from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from find_tunes.core.config import DATABASE_URL


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Song(Base):
    __tablename__ = "songs"
  
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    artist = Column(String, index=True)
    youtube_url = Column(String, unique=True, nullable=False)
    duration = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    fingerprints = relationship("Fingerprint", back_populates="song", cascade="all, delete-orphan") #added cascade="all, delete-orphan" to Song model, deleting the parent Song will automatically trigger PostgreSQL to hunt down and delete any attached fingerprints or vectors
    spec_embeddings = relationship("SpectrogramEmbedding", back_populates="song", cascade="all, delete-orphan")
    pitch_embeddings = relationship("PitchEmbedding", back_populates="song", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Song(title={self.title}, artist={self.artist})>"

class SpectrogramEmbedding(Base):
    __tablename__ = "spectrogram_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False)
    offset = Column(Float, nullable=False) # Time in seconds
    
    # 128-dimensional vector matching your Siamese Network output
    embedding = Column(Vector(128), nullable=False) 
    
    song = relationship("Song", back_populates="spec_embeddings")

class PitchEmbedding(Base):
    __tablename__ = "pitch_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False)
    offset = Column(Float, nullable=False) # Time in seconds
    
    # 128-dimensional vector matching your CRNN output
    embedding = Column(Vector(128), nullable=False)
    
    song = relationship("Song", back_populates="pitch_embeddings")


class Fingerprint(Base):
    __tablename__ = "fingerprints"

    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False)
    
    # The SHA1 Hash (Length 64 to be safe, with index=True for O(1) lookups)
    hash_string = Column(String(64), index=True, nullable=False)
    
    # The Time Offset
    offset = Column(Integer, nullable=False)
    
    song = relationship("Song", back_populates="fingerprints")

    def __repr__(self):
        return f"<Fingerprint(hash={self.hash_string[:10]}..., offset={self.offset})>" 

# Initialize tables
def init_db():
    Base.metadata.create_all(bind=engine)
    
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()