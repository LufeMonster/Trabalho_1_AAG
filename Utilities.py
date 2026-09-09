import time

class Utilities:
    @staticmethod
    def log(msg):
            """Imprime mensagens de progresso com timestamp relativo simples."""
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    @staticmethod
    def timeit(func):
        """Decorator simples para medir e reportar o tempo de cada etapa."""
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            dt = time.time() - t0
            Utilities.log(f"  -> '{func.__name__}' concluída em {dt:.2f}s")
            return result
        return wrapper