class ServiceError(Exception):
    """Base for domain errors mapped to HTTP responses by handlers in app/main.py."""


class InvalidImageError(ServiceError):
    def __init__(self, code, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class EmbeddingError(ServiceError):
    def __init__(self, code, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ModelNotReadyError(ServiceError):
    def __init__(self, message: str = "El modelo todavía no está listo."):
        self.message = message
        super().__init__(message)
