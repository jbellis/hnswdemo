from typing import Any, Dict, List
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

class DB:
    def __init__(self, keyspace: str, table: str, **kwargs):
        self.keyspace = keyspace
        self.table = table
        self.cluster = Cluster(**kwargs)
        self.session = self.cluster.connect()

        # Create keyspace if not exists
        self.session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {keyspace}
            WITH REPLICATION = {{ 'class': 'SimpleStrategy', 'replication_factor': 1 }}
            """
        )

        # Create table if not exists
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {keyspace}.{table} (
            pk int PRIMARY KEY,
            val vector<float, 128>);
            """
        )

        # Create SAI index if not exists
        sai_index_name = f"{table}_embedding_idx"
        self.session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {sai_index_name} ON {keyspace}.{table} (val)
            USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
            """
        )

    def upsert_one(self, pk, embedding):
            query = SimpleStatement(
                f"""
                INSERT INTO {self.keyspace}.{self.table}
                (pk, val)
                VALUES (%s, %s)
                """
            )
            self.session.execute(query, (pk, embedding))

    def query(self, vector: List[float], top_k: int) -> List[str]:
        query = SimpleStatement(
            f"SELECT pk FROM {self.keyspace}.{self.table} ORDER BY val ANN OF %s LIMIT %s"
        )
        res = self.session.execute(query, (vector, top_k))
        rows = [row for row in res]
        # print('\n'.join(repr(row) for row in rows))
        return [row.pk for row in rows]
