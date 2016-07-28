"""Iterators and Generators.

"""

#1 手动遍历迭代器
def manual_iter():
    with open("somefile.txt") as f:
        try:
            while True:
                line = next(f)
                print(line, end="")
        except StopIteration:
            pass
manual_iter()

def manual_iter_new():
    with open("somefile.txt") as f:
            while True:
                line = next(f, None)
                if line is None:
                    break
                print(line, end="")
manual_iter_new()

items = [1, 2, 3]
it = iter(items)
print(next(it))
print(next(it))
print(next(it))

#2 代理迭代
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return "Node({!r})".format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)
if "__main__" == __name__:
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)

    for ch in root:
        print(ch)
