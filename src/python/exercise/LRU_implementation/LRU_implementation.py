from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        # self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move the key to the end to show it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update the value and mark as recently used
            self.cache.move_to_end(key)
        self.cache[key] = value
        # If capacity exceeded, pop the first (least recently used) item
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

def main():
    lru = LRUCache()
    lru.put(1, 1)
    lru.put(2, 2)
    lru.put(3, 3)
    print(lru.get(2))
    lru.cache.popitem()
    print(lru.cache)

if __name__ == "__main__":
    main()