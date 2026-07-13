import psycopg
import traceback

try:
    print("Connecting...")

    conn = psycopg.connect(
        host="findtunes-db.cnaume6w8w1t.ap-south-1.rds.amazonaws.com",
        port=5432,
        dbname="postgres",
        user="postgres",
        password="your_secure_password123",
        sslmode="require"
    )

    print("✅ Connected!")

    cur = conn.cursor()

    cur.execute("SELECT version();")
    print(cur.fetchone())

    cur.execute("SELECT name FROM pg_available_extensions WHERE name='vector';")
    print(cur.fetchall())

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    print("✅ pgvector enabled!")

    cur.close()
    conn.close()

except Exception:
    traceback.print_exc()