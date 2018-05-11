"""批量插入数据到redis中

"""
import redis

if __name__ == '__main__':
    host = 'localhost'
    port = 6379
    key_prefix = 'abc.xyz.'
    max_size = 500000  # 100M
    #max_size = 1000000  # 200M
    #max_size = 2000000  # 400M
    #max_size = 4000000  # 800M

    conn = redis.Redis(host=host, port=port)
    pipe = conn.pipeline()
    for i in range(max_size):
        pipe.set(key_prefix + str(i), "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz", ex=60*30)
        if i % 1000 == 0:
            print("insert")
            pipe.execute()
