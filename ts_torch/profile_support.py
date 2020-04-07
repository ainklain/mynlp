import builtins

if '__builtin__' not in dir() or not hasattr(builtins, 'profile'):
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
