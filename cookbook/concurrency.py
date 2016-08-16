"""Concurrency.

"""

#1 启动和停止线程
import time
def countdown(n):
    while n > 0:
        print('t0-minus', n)
        n -= 1
        time.sleep(0.01)

from threading import Thread
t = Thread(target=countdown, args=(10,), daemon=True)
t.start()

t.join()

if (t.is_alive()):
    print('still running')
else:
    print('completed')

class CountdownTask:
    def __init__(self, n):
        super(CountdownTask, self).__init__()
        self.n = n
        self._running = True

    def terminate(self):
        self._running = False
    
    def run(self):
        while self._running and self.n > 0:
            print('t1-minus', self.n)
            self.n -= 1
            time.sleep(0.01)

c = CountdownTask(10)
t = Thread(target=c.run)
t.start()
time.sleep(0.4)
c.terminate()
t.join()

class CountdownThread(Thread):
    def __init__(self, n):
        super(CountdownThread, self).__init__()
        self.n = n

    def run(self):
        while self.n > 0:
            print('t2-minus', self.n)
            self.n -= 1
            time.sleep(0.01)

c = CountdownThread(5)
c.start()
c.join()

# 单独的进程中执行
if __name__ == '__main__':
    print('==============')
    import multiprocessing
    countdown = CountdownTask(5)
    p = multiprocessing.Process(target=countdown.run)
    p.start()
    p.join()

#2 判断线程是否已经启动
from threading import Event

# code to execute in an independent thread
def countdown(n, started_evt):
    print("countdown starting")
    started_evt.set() 
    while n > 0:
        print("t3-minus", n)
        n -= 1
        time.sleep(0.01)

# create the event object that will be used to signal startup
started_evt = Event()

# launch the thread and pass the startup event
print("launching countdown")
t = Thread(target=countdown, args=(10, started_evt))
t.start()

# wait for the thread to start
started_evt.wait()
print("countdown is running")
t.join()

from threading import Condition

class PeriodicTimer:
    def __init__(self, interval):
        self._interval = interval
        self._flag = 0
        self._cv = Condition()

    def start(self):
        t = Thread(target=self.run)
        t.daemon = True
        t.start()

    def run(self):
        """Run the timer and notify waiting threads after each interval"""
        while True:
            time.sleep(self._interval)
            with self._cv:
                 self._flag ^= 1
                 self._cv.notify_all() # 唤醒所有等待的线程

    def wait_for_tick(self):
        """Wait for the next tick of the timer"""
        with self._cv:
            last_flag = self._flag
            while last_flag == self._flag:
                self._cv.wait()

ptimer = PeriodicTimer(0.02)
ptimer.start()

# two threads that synchronize on the timer
def countdown(nticks):
    while nticks > 0:
        ptimer.wait_for_tick()
        print("t4-minus", nticks)
        nticks -= 1

def countup(last):
    n = 0
    while n < last:
        ptimer.wait_for_tick()
        print("counting", n)
        n += 1

t1 = Thread(target=countdown, args=(10,))
t1.start()
t2 = Thread(target=countup, args=(5,))
t2.start()

t1.join()
t2.join()

# 唤醒一个单独的等待线程
import threading

def worker(n, semaphore):
    # wait to be signaled
    semaphore.acquire()
    print('working', n)

semaphore = threading.Semaphore(0)
nworkers = 10
for n in range(nworkers):
    t = threading.Thread(target=worker, args=(n, semaphore,))
    t.start()

for n in range(nworkers):
    semaphore.release()

#3 线程间通信
from queue import Queue

# object that signals shutdown
_sentinel = object()

# a thread that produces data
def producer(out_q):
    n = 10
    while n > 0:
        # Produce some data
        out_q.put(n)
        time.sleep(0.02)
        n -= 1

    # put the sentinel on the queue to indicate completion
    out_q.put(_sentinel)

# a thread that consumes data
def consumer(in_q):
    while True:
        # get some data
        data = in_q.get()

        # check for termination
        if data is _sentinel:
            in_q.put(_sentinel)
            break

        # process the data
        print('got:', data)
    print('consumer shutting down')

if __name__ == '__main__':
    q = Queue()
    t1 = Thread(target=consumer, args=(q,))
    t2 = Thread(target=producer, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# 线程安全的优先级队列
import heapq, queue

class PriorityQueue(object):
    def __init__(self):
        super(PriorityQueue, self).__init__()
        self._queue = []
        self._count = 0
        self._cv = threading.Condition()

    def put(self, item, priority):
        with self._cv:
            heapq.heappush(self._queue, (-priority, self._count, item))
            self._count += 1
            self._cv.notify()

    def get(self):
        with self._cv:
            while len(self._queue) == 0:
                self._cv.wait()
            return heapq.heappop(self._queue)[-1]

def producer(q):
    print('producing items')
    q.put('C', 5)
    q.put('A', 15)
    q.put('B', 10)
    q.put('D', 0)
    q.put(None, -100)

def consumer(q):
    time.sleep(0.1)
    print('getting items')
    while True:
        item = q.get()
        if item is None:
            break
        print('got:', item)
    print('consumer done')

if __name__ == '__main__':
    q = PriorityQueue()
    t1 = threading.Thread(target=producer, args=(q,))
    t2 = threading.Thread(target=consumer, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# 事件完成功能
def producer(out_q):
    n = 10
    while n > 0:
        out_q.put(n)
        time.sleep(0.02)
        n -= 1

def consumer(in_q):
    while True:
        try:
            data = in_q.get(timeout=1)
        except queue.Empty as e:
            break
        print('got:', data)
        # indicate completion
        in_q.task_done()
    print('consumer shutting down')

if __name__ == '__main__':
    q = Queue()
    t1 = Thread(target=consumer, args=(q,))
    t2 = Thread(target=producer, args=(q,))
    t1.start()
    t2.start()
    # wait for all produced items to be consumed
    q.join()

# 事件感知
def producer(out_q):
    n = 10
    while n > 0:
        event = Event()
        out_q.put((n, event))
        event.wait()
        time.sleep(0.02)
        n -= 1

def consumer(in_q):
    while True:
        try:
            data, event = in_q.get(timeout=1)
            event.set()
        except queue.Empty as e:
            break
        print('got:', data)
    print('consumer shutting down')

if __name__ == '__main__':
    q = Queue()
    t1 = Thread(target=consumer, args=(q,))
    t2 = Thread(target=producer, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

#4 对临界区加锁
class SharedCounter:
    """a counter object that can be shared by multiple threads."""
    def __init__(self, initial_value=0):
        super(SharedCounter, self).__init__()
        self._value = initial_value
        self._value_lock = threading.Lock()

    def incr(self, delta=1):
        with self._value_lock:
            self._value += delta

    def decr(self, delta=1):
        with self._value_lock:
            self._value -= delta

class SharedCounter:
    """a counter object that can be shared by multiple threads."""
    _lock = threading.RLock() # 可重入锁
    def __init__(self, initial_value=0):
        super(SharedCounter, self).__init__()
        self._value = initial_value

    def incr(self, delta=1):
        with SharedCounter._lock:
            self._value += delta

    def decr(self, delta=1):
        with SharedCounter._lock:
            self.incr(-delta)

#5 避免死锁
from contextlib import contextmanager

# thread-local state to stored information on locks already acquired
_local = threading.local()

@contextmanager
def acquire(*locks):
    # sort locks by object identifier
    locks = sorted(locks, key=lambda x: id(x))

    # make sure lock order of previously acquired locks is not violated
    acquired = getattr(_local, 'acquired', [])
    if acquired and max(id(lock) for lock in acquired) >= id(locks[0]):
        raise RuntimeError('Lock Order Violation')

    # acquire all of the locks
    acquired.extend(locks)
    _local.acquired = acquired
    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        # release locks in reverse order of acquisition
        for lock in reversed(locks):
            lock.release()
        del acquired[-len(locks):]

x_lock = threading.Lock()
y_lock = threading.Lock()

def thread_1():
    while True:
        with acquire(x_lock, y_lock):
            print('thread-1')

def thread_2():
    while True:
        with acquire(y_lock, x_lock):
            print('thread-2')
t1 = threading.Thread(target=thread_1)
t1.daemon = True
# t1.start()

t2 = threading.Thread(target=thread_2)
t2.daemon = True
# t2.start()

# 可能产生死锁
def thread_1():
    while True:
        with acquire(x_lock):
            with acquire(y_lock):
                print('thread-1')

def thread_2():
    while True:
        with acquire(y_lock):
            with acquire(x_lock):
                print('thread-2')
t1 = threading.Thread(target=thread_1)
t1.daemon = True
# t1.start()

t2 = threading.Thread(target=thread_2)
t2.daemon = True
# t2.start()

# 哲学家就餐问题
def philosopher(left, right):
    n = 0
    while True:
        with acquire(left, right):
            n += 1
            if n == 10:
                break
            print(threading.currentThread(), 'eating')

NSTICKS = 5
chopsticks = [threading.Lock() for n in range(NSTICKS)]
for n in range(NSTICKS):
    t = threading.Thread(target=philosopher, args=(chopsticks[n], chopsticks[(n+1) % NSTICKS]))
    t.daemon = True
    t.start()
    t.join()

#6 保存线程专有状态
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = AF_INET
        self.type = SOCK_STREAM
        self.local = threading.local()

    def __enter__(self):
        if hasattr(self.local, 'sock'):
            raise RuntimeError('Already connected')
        self.local.sock = socket(self.family, self.type)
        self.local.sock.connect(self.address)
        return self.local.sock
        
    def __exit__(self, exc_ty, exc_val, tb):
        self.local.sock.close()
        del self.local.sock

def test(conn):
    from functools import partial

    # connection closed
    with conn as s:
        # conn.__enter__() executes: connection open
        s.send(b'GET /index.html HTTP/1.0\r\n')
        s.send(b'Host: www.python.org\r\n')
        s.send(b'\r\n')
        resp = b''.join(iter(partial(s.recv, 8192), b''))
        # conn.__exit__() executes: connection closed

    print('got {} bytes'.format(len(resp)))

if __name__ == '__main__':
    conn = LazyConnection(('www.python.org', 80))

    t1 = threading.Thread(target=test, args=(conn,))
    t2 = threading.Thread(target=test, args=(conn,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = AF_INET
        self.type = SOCK_STREAM
        self.local = threading.local()

    def __enter__(self):
        sock = socket(self.family, self.type)
        sock.connect(self.address)
        if not hasattr(self.local, 'connections'):
            self.local.connections = []
        self.local.connections.append(sock)
        return sock
                
    def __exit__(self, exc_ty, exc_val, tb):
        self.local.connections.pop().close()

def test(conn):
    # Example use
    from functools import partial

    with conn as s:
        s.send(b'GET /index.html HTTP/1.0\r\n')
        s.send(b'Host: www.python.org\r\n')
        s.send(b'\r\n')
        resp = b''.join(iter(partial(s.recv, 8192), b''))

    print('Got {} bytes'.format(len(resp)))

    with conn as s1, conn as s2:
        s1.send(b'GET /downloads HTTP/1.0\r\n')
        s2.send(b'GET /index.html HTTP/1.0\r\n')
        s1.send(b'Host: www.python.org\r\n')
        s2.send(b'Host: www.python.org\r\n')
        s1.send(b'\r\n')
        s2.send(b'\r\n')
        resp1 = b''.join(iter(partial(s1.recv, 8192), b''))
        resp2 = b''.join(iter(partial(s2.recv, 8192), b''))

    print('resp1 got {} bytes'.format(len(resp1)))
    print('resp2 got {} bytes'.format(len(resp2)))

if __name__ == '__main__':
    conn = LazyConnection(('www.python.org', 80))
    t1 = threading.Thread(target=test, args=(conn,))
    t2 = threading.Thread(target=test, args=(conn,))
    t3 = threading.Thread(target=test, args=(conn,))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()

#7 创建线程池
from concurrent.futures import ThreadPoolExecutor

def echo_client(sock, client_addr):
    """handle a client connection"""
    print('got connection from', client_addr)
    while True:
        msg = sock.recv(65536)
        if not msg:
            break
        sock.sendall(msg)
    print('client closed connection')
    sock.close()

def echo_server(addr):
    print('echo server running at', addr)
    pool = ThreadPoolExecutor(128)
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(addr)
    sock.listen(5)
    while True:
        client_sock, client_addr = sock.accept()
        pool.submit(echo_client, client_sock, client_addr)

# echo_server(('',15000))

import urllib.request

def fetch_url(url):
    u = urllib.request.urlopen(url)
    data = u.read()
    return data

pool = ThreadPoolExecutor(10)
# submit work to the pool
a = pool.submit(fetch_url, 'http://www.python.org')
b = pool.submit(fetch_url, 'http://www.pypy.org')
# get the results back
x = a.result()
y = a.result()

#8 实现简单的并行编程
import gzip
import io
import glob

def find_robots(filename):
    """find all of the hosts that access robots.txt in a single log file"""
    robots = set()
    with gzip.open(filename) as f:
        for line in io.TextIOWrapper(f, encoding='ascii'):
            fields = line.split()
            if fields[6] == '/robots.txt':
                robots.add(fields[0])
    return robots

def find_all_robots(logdir):
    """find all hosts across and entire sequence of files"""
    files = glob.glob(logdir+"/*.log.gz")
    all_robots = set()
    for robots in map(find_robots, files):
        all_robots.update(robots)
    return all_robots

if __name__ == '__main__':
    import time
    start = time.time()
    robots = find_all_robots("data/concurrent/logs")
    end = time.time()
    for ipaddr in robots:
        print(ipaddr)
    print('Took {:f} seconds'.format(end-start))

from concurrent import futures

def find_all_robots(logdir):
    """find all hosts across and entire sequence of files"""
    files = glob.glob(logdir+"/*.log.gz")
    all_robots = set()
    with futures.ProcessPoolExecutor() as pool:
        for robots in pool.map(find_robots, files):
            all_robots.update(robots)
    return all_robots

if __name__ == '__main__':
    import time
    start = time.time()
    robots = find_all_robots("data/concurrent/logs")
    end = time.time()
    for ipaddr in robots:
        print(ipaddr)
    print('Took {:f} seconds'.format(end-start))

#9 如何规避GIL带来的限制
# processing pool
pool = None

def some_work(args):
    result = "performs a large calculation (CPU bound)"
    return result

def some_thread():
    while True:
        r = pool.apply(some_work, (args))

# initialize the pool
if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool()

#10 定义一个Actor任务
# sentinel used for shutdown
class ActorExit(Exception):
    pass

class Actor:
    def __init__(self):
        self._mailbox = Queue()

    def send(self, msg):
        self._mailbox.put(msg)

    def recv(self):
        msg = self._mailbox.get()
        if msg is ActorExit:
            raise ActorExit()
        return msg
    
    def close(self):
        self.send(ActorExit)

    def start(self):
        self._terminated = Event()
        t = Thread(target=self._bootstrap)
        t.daemon = True
        t.start()

    def _bootstrap(self):
        try:
            self.run()
        except ActorExit:
            pass
        finally:
            self._terminated.set()

    def join(self):
        self._terminated.wait()

    def run(self):
        while True:
            msg = self.recv()

class PrintActor(Actor):
    def run(self):
        while True:
            msg = self.recv()
            print('got:', msg)

if __name__ == '__main__':
    p = PrintActor()
    p.start()
    p.send('hello')
    p.send('world')
    p.close()
    p.join()

class TaggedActor(Actor):
    def run(self):
        while True:
             tag, *payload = self.recv()
             getattr(self, 'do_' + tag)(*payload)
    
    # methods correponding to different message tags
    def do_A(self, x):
        print("running A", x)

    def do_B(self, x, y):
        print("running B", x, y)

if __name__ == '__main__':
    a = TaggedActor()
    a.start()
    a.send(('A', 1)) # invokes do_A(1)
    a.send(('B', 2, 3)) # invokes do_B(2,3)
    a.close()
    a.join()

class Result:
    def __init__(self):
        self._evt = Event()
        self._result = None

    def set_result(self, value):
        self._result = value
        self._evt.set()

    def result(self):
        self._evt.wait()
        return self._result

class Worker(Actor):
    def submit(self, func, *args, **kwargs):
        r = Result()
        self.send((func, args, kwargs, r))
        return r

    def run(self):
        while True:
            func, args, kwargs, r = self.recv()
            r.set_result(func(*args, **kwargs))

if __name__ == '__main__':
    worker = Worker()
    worker.start()
    r = worker.submit(pow, 2, 3)
    print(r.result())
    worker.close()
    worker.join()

#11 实现发布者/订阅者消息模式
from collections import defaultdict

class Exchange:
    def __init__(self):
        self._subscribers = set()

    def attach(self, task):
        self._subscribers.add(task)

    def detach(self, task):
        self._subscribers.remove(task)

    def send(self, msg):
        for subscriber in self._subscribers:
            subscriber.send(msg)

# dictionary of all created exchanges
_exchanges = defaultdict(Exchange)

# return the Exchange instance associated with a given name
def get_exchange(name):
    return _exchanges[name]

if __name__ == '__main__':
    # example task (just for testing)
    class Task:
        def __init__(self, name):
            self.name = name

        def send(self, msg):
            print('{} got: {!r}'.format(self.name, msg))

    task_a = Task('A')
    task_b = Task('B')

    exc = get_exchange('spam')
    exc.attach(task_a)
    exc.attach(task_b)
    exc.send('msg1')
    exc.send('msg2')

    exc.detach(task_a)
    exc.detach(task_b)
    exc.send('msg3')

class Exchange:
    def __init__(self):
        self._subscribers = set()

    def attach(self, task):
        self._subscribers.add(task)

    def detach(self, task):
        self._subscribers.remove(task)

    @contextmanager
    def subscribe(self, *tasks):
        for task in tasks:
            self.attach(task)
        try:
            yield
        finally:
            for task in tasks:
                self.detach(task)

    def send(self, msg):
        for subscriber in self._subscribers:
            subscriber.send(msg)

# dictionary of all created exchanges
_exchanges = defaultdict(Exchange)

# return the Exchange instance associated with a given name
def get_exchange(name):
    return _exchanges[name]

if __name__ == '__main__':
    # example task (just for testing)
    class Task:
        def __init__(self, name):
            self.name = name

        def send(self, msg):
            print('{} got: {!r}'.format(self.name, msg))

    task_a = Task('A')
    task_b = Task('B')

    exc = get_exchange('spam')
    with exc.subscribe(task_a, task_b):
        exc.send('msg1')
        exc.send('msg2')

    exc.send('msg3')

#12 使用生成器作为线程的替代方案
# a very simple example of a coroutine/generator scheduler
# two simple generator functions
def countdown(n):
    while n > 0:
        print('t-minus', n)
        yield
        n -= 1
    print('blastoff!')

def countup(n):
    x = 0
    while x < n:
        print('counting up', x)
        yield
        x += 1

from collections import deque

class TaskScheduler:
    def __init__(self):
        self._task_queue = deque()

    def new_task(self, task):
        """admit a newly started task to the scheduler"""
        self._task_queue.append(task)

    def run(self):
        """run until there are no more tasks"""
        while self._task_queue:
            task = self._task_queue.popleft()
            try:
                # run until the next yield statement
                next(task)
                self._task_queue.append(task)
            except StopIteration:
                # generator is no longer executing
                pass

sched = TaskScheduler()
sched.new_task(countdown(10))
sched.new_task(countdown(5))
sched.new_task(countup(15))
sched.run()

#13 轮询多个线程队列

#14 在UNIX上加载守护进程
