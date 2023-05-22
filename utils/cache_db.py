import threading
import numpy as np
from typing import Optional
from peewee import Model, SqliteDatabase, CharField, BlobField, AutoField


thread_pool = []
thread_pool_limit = 512
db_mutex = threading.Lock()
database = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = database


class CacheEntry(BaseModel):
    id = AutoField()
    key = CharField(max_length=31, index=True, unique=True, null=False, column_name='key')
    logit = BlobField(null=False, column_name='logit')


def init_db(path: str):
    database.init(path)
    database.connection()
    database.create_tables([CacheEntry])


def write_cache_thread(split: str, sample_key: str, logit: np.ndarray, dtype):
    key, logit_bin = '-'.join((split, sample_key)), logit.astype(dtype).tobytes()
    db_mutex.acquire()
    CacheEntry.insert(key=key, logit=logit_bin).on_conflict_replace().execute()
    db_mutex.release()


def write_cache(split: str, sample_key: str, logit: np.ndarray, dtype):
    cur_thread = threading.Thread(target=write_cache_thread, args=(split, sample_key, logit, dtype))
    cur_thread.start()
    thread_pool.append(cur_thread)
    if len(thread_pool) == thread_pool_limit:
        clear_write_thread()


def clear_write_thread():
    # print('joining all threads ......')
    for thread in thread_pool:
        thread.join()
    thread_pool.clear()


def read_cache(split: str, sample_key: str, dtype) -> Optional[np.ndarray]:
    key = '-'.join((split, sample_key))
    db_mutex.acquire()
    ret = CacheEntry.select().where(CacheEntry.key == key)
    ret = [item for item in ret]
    db_mutex.release()
    if len(ret) == 0:
        return None
    assert len(ret) == 1, key
    logit = np.frombuffer(ret[0].logit, dtype=dtype)
    return logit


def reset_database(path: str, from_split: int, to_split: int):
    init_db(path)
    for sid in range(from_split, to_split):
        prefix_key = f'p{sid}-'
        CacheEntry.delete().where(CacheEntry.key.startswith(prefix_key)).execute()
