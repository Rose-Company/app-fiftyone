class CustomError(Exception):
    def __init__(
        self, 
        message: str, 
        code: int = None, 
        internal_message: str = None
    ):

        self.message = message
        self.status = code
        self.internal_message = internal_message
        super().__init__(self.message)

    def __str__(self):
        if self.status:
            return f"{self.message}"
        return self.message

class AIError(Exception):
    def __init__(
        self, 
        message: str, 
        code: int = None, 
        internal_message: str = None, 
        ai_raw_response: str = None
    ):

        self.message = message
        self.status = code
        self.internal_message = internal_message
        self.ai_raw_response = ai_raw_response 
        super().__init__(self.message)

    def __str__(self):
        if self.status:
            return f"{self.message}"
        return self.message
    
invalid_input_error = CustomError(
    "Invalid input data provided", 400, "")
invalid_task_type = CustomError(
    "Invalid task type provided", 400, "")
invalid_writing_essay = CustomError(
    "Invalid writing essay", 400, "")
