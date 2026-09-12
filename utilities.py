import time


class Utilities:
    def __init__(self):
        pass

    @staticmethod
    def log(msg):
        """Prints progress messages with a simple relative timestamp."""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    @staticmethod
    def timeit(func):
        """Simple decorator that measures and reports the runtime of each step."""
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            dt = time.time() - t0
            Utilities.log(f"  -> '{func.__name__}' finished in {dt:.2f}s")
            return result
        return wrapper
