"""
context/conversation_db.py -- Professional SQLite-backed conversation memory.

Features:
- Thread-safe SQLite connection (WAL mode enabled).
- FTS5 integration for semantic retrieval.
- Automated schema migrations.
- Comprehensive logging and type safety.
"""

import hashlib
import logging
import os
import sqlite3
import time
from typing import List, Dict, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data", 
    "conversations.db"
)


class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass


class ConversationDB:
    """
    Manages long-term memory and state for the WhatsApp agent using SQLite.
    
    Provides interfaces for message storage, FTS search, and summary management.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialise the database connection and ensure schema is ready.
        
        Args:
            db_path: Absolute path to the SQLite database file.
        """
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        try:
            # check_same_thread=False is safe for our sequential agent loop
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._set_pragmas()
            self._init_schema()
            logger.info(f"Database initialised at {self.db_path}")
        except sqlite3.Error as e:
            logger.critical(f"Failed to connect to database: {e}")
            raise DatabaseError(f"Database initialisation failed: {e}") from e

    def _set_pragmas(self):
        """Optimise SQLite performance with recommended settings."""
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-4000")  # 4MB cache
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def _init_schema(self):
        """Create tables, indexes, and triggers if they don't exist."""
        try:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS contacts (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT    UNIQUE NOT NULL,
                    created_at REAL    NOT NULL DEFAULT (unixepoch('now'))
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact_id  INTEGER NOT NULL REFERENCES contacts(id),
                    role        TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
                    content     TEXT    NOT NULL,
                    content_hash TEXT,
                    intent      TEXT,
                    emotion     TEXT,
                    confidence  REAL,
                    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_messages_contact
                    ON messages(contact_id, created_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_messages_hash
                    ON messages(contact_id, content_hash);

                CREATE TABLE IF NOT EXISTS conversation_summary (
                    contact_id  INTEGER PRIMARY KEY REFERENCES contacts(id),
                    summary     TEXT    NOT NULL,
                    msg_count   INTEGER NOT NULL DEFAULT 0,
                    updated_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
                );

                CREATE TABLE IF NOT EXISTS agent_state (
                    contact_id  INTEGER NOT NULL REFERENCES contacts(id),
                    key         TEXT    NOT NULL,
                    value       TEXT,
                    PRIMARY KEY (contact_id, key)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='id',
                    tokenize='porter unicode61'
                );

                -- Keep FTS5 index in sync
                CREATE TRIGGER IF NOT EXISTS messages_ai
                  AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
                  END;

                CREATE TRIGGER IF NOT EXISTS messages_ad
                  AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                  END;

                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject     TEXT    NOT NULL,
                    predicate   TEXT    NOT NULL,
                    object      TEXT    NOT NULL,
                    created_at  REAL    NOT NULL DEFAULT (unixepoch('now')),
                    UNIQUE(subject, predicate, object)
                );
            """)
            self._conn.commit()
            self._run_migrations()
        except sqlite3.Error as e:
            logger.error(f"Schema initialisation failed: {e}")
            raise DatabaseError(f"Schema error: {e}") from e

    def _run_migrations(self):
        """Handle incremental schema updates."""
        # Check if content_hash column exists (for older databases)
        cursor = self._conn.execute("PRAGMA table_info(messages)")
        cols = {r[1] for r in cursor.fetchall()}
        
        if "content_hash" not in cols:
            logger.info("Migrating database: adding content_hash column to messages.")
            self._conn.execute("ALTER TABLE messages ADD COLUMN content_hash TEXT")
            self._conn.commit()
            # Backfill hashes
            cursor = self._conn.execute("SELECT id, content FROM messages")
            for mid, content in cursor.fetchall():
                h = self._compute_hash(content)
                self._conn.execute("UPDATE messages SET content_hash=? WHERE id=?", (h, mid))
            self._conn.commit()

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute MD5 hash for deduplication."""
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    # ------------------------------------------------------------------ #
    # Interface Methods                                                    #
    # ------------------------------------------------------------------ #

    def get_or_create_contact(self, name: str) -> int:
        """Get existing contact ID or create a new one."""
        try:
            row = self._conn.execute("SELECT id FROM contacts WHERE name = ?", (name,)).fetchone()
            if row:
                return row[0]
            
            cur = self._conn.execute("INSERT INTO contacts (name) VALUES (?)", (name,))
            self._conn.commit()
            return cur.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to get/create contact '{name}': {e}")
            return -1

    def add_message(
        self,
        contact_id: int,
        role: str,
        content: str,
        intent: Optional[str] = None,
        emotion: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Store a new message in the conversation history."""
        try:
            content_hash = self._compute_hash(content)
            cur = self._conn.execute(
                "INSERT INTO messages (contact_id, role, content, content_hash, intent, emotion, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (contact_id, role, content, content_hash, intent, emotion, confidence)
            )
            self._conn.commit()
            return cur.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to add message for contact {contact_id}: {e}")
            return -1

    def has_message(self, contact_id: int, content: str) -> bool:
        """Check for existence of a message to prevent duplicates."""
        h = self._compute_hash(content)
        row = self._conn.execute(
            "SELECT 1 FROM messages WHERE contact_id=? AND content_hash=? LIMIT 1",
            (contact_id, h)
        ).fetchone()
        return row is not None

    def get_recent_messages(self, contact_id: int, limit: int = 12) -> List[Dict[str, str]]:
        """Fetch the most recent N messages, returned in chronological order."""
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE contact_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (contact_id, limit)
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def get_time_since_last_message(self, contact_id: int) -> float:
        """Returns the number of seconds since the last message in this chat."""
        try:
            row = self._conn.execute(
                "SELECT strftime('%s', 'now') - strftime('%s', created_at) "
                "FROM messages WHERE contact_id = ? ORDER BY created_at DESC LIMIT 1",
                (contact_id,)
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0
        except sqlite3.Error as e:
            logger.error(f"Failed to get time since last message for {contact_id}: {e}")
            return 0.0

    def search_relevant(self, contact_id: int, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """Perform semantic search using SQLite FTS5."""
        if not query.strip():
            return []
            
        # Basic sanitisation for FTS5
        sanitized_query = '"' + " ".join(query.split()[:10]).replace('"', '') + '"'
        try:
            query_str = (
                "SELECT m.role, m.content FROM messages_fts f "
                "JOIN messages m ON f.rowid = m.id "
                f"WHERE m.contact_id = ? AND messages_fts MATCH ? "
                "ORDER BY rank LIMIT ?"
            )
            rows = self._conn.execute(query_str, (contact_id, sanitized_query, limit)).fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]
        except sqlite3.Error as e:
            logger.warning(f"FTS search failed for query '{query}': {e}")
            return []

    # ------------------------------------------------------------------ #
    # Knowledge Graph                                                      #
    # ------------------------------------------------------------------ #

    def add_knowledge_triple(self, subject: str, predicate: str, obj: str):
        """Insert a relationship triple into the knowledge graph."""
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO knowledge_graph (subject, predicate, object) VALUES (?, ?, ?)",
                (subject.strip(), predicate.strip(), obj.strip())
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to add triple ({subject}, {predicate}, {obj}): {e}")

    def query_knowledge_graph(self, entity: str) -> List[Dict[str, str]]:
        """Find all triples related to a specific entity."""
        try:
            rows = self._conn.execute(
                "SELECT subject, predicate, object FROM knowledge_graph "
                "WHERE subject LIKE ? OR object LIKE ?",
                (f"%{entity}%", f"%{entity}%")
            ).fetchall()
            return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Knowledge graph query failed for '{entity}': {e}")
            return []

    def get_full_knowledge_graph(self) -> List[Dict[str, str]]:
        """Fetch the entire knowledge graph for visualization."""
        try:
            rows = self._conn.execute(
                "SELECT subject, predicate, object FROM knowledge_graph"
            ).fetchall()
            return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch full knowledge graph: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Summary & State Management                                           #
    # ------------------------------------------------------------------ #

    def save_summary(self, contact_id: int, summary: str):
        """Update the conversation summary for a contact."""
        msg_count = self.get_message_count(contact_id)
        try:
            self._conn.execute(
                "INSERT INTO conversation_summary (contact_id, summary, msg_count, updated_at) "
                "VALUES (?, ?, ?, unixepoch('now')) "
                "ON CONFLICT(contact_id) DO UPDATE SET "
                "summary=excluded.summary, msg_count=excluded.msg_count, updated_at=excluded.updated_at",
                (contact_id, summary, msg_count)
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save summary for {contact_id}: {e}")

    def get_summary(self, contact_id: int) -> Optional[str]:
        """Retrieve current summary for a contact."""
        row = self._conn.execute(
            "SELECT summary FROM conversation_summary WHERE contact_id = ?", 
            (contact_id,)
        ).fetchone()
        return row[0] if row else None

    def get_message_count(self, contact_id: int) -> int:
        """Total count of messages for a contact."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE contact_id = ?", 
            (contact_id,)
        ).fetchone()
        return row[0] if row else 0

    def summary_needs_update(self, contact_id: int, threshold: int = 20) -> bool:
        """Check if enough new messages exist to warrant a summary refresh."""
        row = self._conn.execute(
            "SELECT msg_count FROM conversation_summary WHERE contact_id = ?", 
            (contact_id,)
        ).fetchone()
        if not row:
            return True
        current = self.get_message_count(contact_id)
        return (current - row[0]) >= threshold

    def set_state(self, contact_id: int, key: str, value: Any):
        """Store arbitrary agent state for a contact."""
        try:
            self._conn.execute(
                "INSERT INTO agent_state (contact_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(contact_id, key) DO UPDATE SET value=excluded.value",
                (contact_id, key, str(value))
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to set state {key} for {contact_id}: {e}")

    def get_state(self, contact_id: int, key: str) -> Optional[str]:
        """Retrieve stored agent state."""
        row = self._conn.execute(
            "SELECT value FROM agent_state WHERE contact_id=? AND key=?", 
            (contact_id, key)
        ).fetchone()
        return row[0] if row else None

    def get_contacts_by_category(self, category: str) -> List[str]:
        """
        Retrieve a list of contact names that belong to a specific relationship category.
        This provides the 'multi-list' functionality requested by the user.
        """
        try:
            rows = self._conn.execute(
                "SELECT c.name FROM contacts c "
                "JOIN agent_state a ON c.id = a.contact_id "
                "WHERE a.key = 'relationship_category' AND a.value = ?",
                (category,)
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get contacts for category '{category}': {e}")
            return []

    def get_last_message_time(self, contact_id: int) -> Optional[float]:
        """Timestamp of the most recent message."""
        row = self._conn.execute(
            "SELECT MAX(created_at) FROM messages WHERE contact_id = ?", 
            (contact_id,)
        ).fetchone()
        return row[0] if row and row[0] else None

    def get_proactive_count_today(self, contact_id: int) -> int:
        """Count proactive messages sent in the current calendar day."""
        # Start of current day (00:00:00)
        start_of_day = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages "
            "WHERE contact_id=? AND role='assistant' AND intent='proactive' AND created_at >= ?",
            (contact_id, start_of_day)
        ).fetchone()
        return row[0] if row else 0

    def build_llm_context(self, contact_id: int, incoming_text: str) -> List[Dict[str, str]]:
        """
        Assemble the full LLM context:
        1. Historical summary
        2. FTS-retrieved relevant older context
        3. Recent message window
        """
        context: List[Dict[str, str]] = []
        
        # 1. Summary
        summary = self.get_summary(contact_id)
        if summary:
            context.append({"role": "system", "content": f"## CONVERSATION SUMMARY:\n{summary}"})
            
        # 2. Relevant Context
        recent = self.get_recent_messages(contact_id, limit=12)
        recent_contents = {m["content"] for m in recent}
        
        if incoming_text.strip():
            relevant = self.search_relevant(contact_id, incoming_text, limit=3)
            # Dedup against recent window
            filtered_relevant = [m for m in relevant if m["content"] not in recent_contents]
            if filtered_relevant:
                formatted = "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in filtered_relevant)
                context.append({"role": "system", "content": f"## RELEVANT PAST CONTEXT:\n{formatted}"})
                
        # 3. Recent window
        context.extend(recent)
        return context

    def get_all_messages_for_summary(self, contact_id: int, limit: int = 200) -> List[Dict[str, str]]:
        """Fetch historical messages for summary generation."""
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE contact_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (contact_id, limit)
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
