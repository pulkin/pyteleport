class NULLMeta(type):
    def __str__(cls):
        return "<NULL>"

    def __repr__(cls):
        return "NULL"


class NULL(metaclass=NULLMeta):
    """Represents NULL"""
    def __new__(cls):
        return cls

