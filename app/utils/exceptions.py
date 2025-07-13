class ValidationError(Exception):
    """Exception raised when validation fails"""
    def __init__(self, errors):
        if isinstance(errors, list):
            self.message = "; ".join(errors)
            self.errors = errors
        else:
            self.message = str(errors)
            self.errors = [str(errors)]
        super().__init__(self.message)

class NotFoundException(Exception):
    """Exception raised when a resource is not found"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class UnauthorizedException(Exception):
    """Exception khi user không có quyền truy cập"""
    pass 

class VideoNotFoundException(Exception):
    """Raised when a video is not found in the database"""
    pass

class InvalidDataFormatException(Exception):
    """Raised when the data format in the JSON file is invalid"""
    pass

class DatabaseConnectionError(Exception):
    """Raised when there is an error connecting to the database"""
    pass

class FileProcessingError(Exception):
    """Raised when there is an error processing a file"""
    pass 