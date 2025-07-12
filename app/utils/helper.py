import re
from typing import Any, Dict
from fastapi import Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.utils.error import CustomError, AIError
import json
from datetime import datetime
from sqlalchemy.ext.declarative import DeclarativeMeta
import traceback
from app.utils import logger


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Add more custom serialization logic here if needed
        return super().default(obj)


def sqlalchemy_to_dict(obj):
    """
    Convert a SQLAlchemy model instance to a dictionary.
    """
    return {column.name: getattr(obj, column.name) for column in obj.__table__.columns}


def response_ok(
    data: Any = None, 
    message: str = "ok", 
    status: int = 200
) -> JSONResponse:
    try:
        logger.info(f"[response_ok] Input data: {data}, message: {message}, status: {status}")
        if isinstance(data, BaseModel):
            data = data.model_dump(by_alias=True)
        elif isinstance(data.__class__, DeclarativeMeta):
            data = sqlalchemy_to_dict(data)

        response = {
            "message": message,
            "data": data
        }
        logger.info(f"[response_ok] Response to jsonify: {response}")
        return JSONResponse(content=response, status_code=status)
    except Exception as e:
        logger.error(f"[response_ok] Serialization error: {e}\n{traceback.format_exc()}\nInput data: {data}")
        return JSONResponse(content={"message": "Internal Server Error", "data": str(e)}, status_code=500)



def response_error(
    error: CustomError, 
    status: int = 500
) -> JSONResponse:

    status = error.status if hasattr(error, 'status') else 500
    response = {
        "status": error.status if hasattr(error, 'status') else 500,
        "message": str(error) if hasattr(error, 'message') else "Đã xảy ra lỗi",
        "internal": error.internal_message if hasattr(error, 'internal_message') else str(error)
    }
    return JSONResponse(content=response, status_code=status)

def ai_response_error(
    error: AIError, 
    status: int = 500
) -> JSONResponse:

    status = error.status if hasattr(error, 'status') else 500
    response = {
        "status": error.status if hasattr(error, 'status') else 500,
        "message": str(error) if hasattr(error, 'message') else "Đã xảy ra lỗi",
        "internal": error.internal_message if hasattr(error, 'internal_message') else str(error),
        "ai_raw_response": error.ai_raw_response if hasattr(error, 'ai_raw_response') else None
    }
    return JSONResponse(content=response, status_code=status)

def format_error_message(
    internal_message: str, 
    ai_raw_response: str
) -> AIError:
    return AIError(
        code=400,
        message="AI's response is not in the correct format",
        internal_message=internal_message, 
        ai_raw_response=ai_raw_response
    )

def connection_error_message(internal_message: str) -> AIError:
    return AIError(
        code=500,
        message="Unable to connect to AI's service",
        internal_message=internal_message, 
    )

def message_ok(
        data: Any = None, 
        message: str = "ok", 
        status: int = 200
    ) -> Dict:

    if isinstance(data, BaseModel):
        data = data.model_dump(by_alias=True)
    elif isinstance(data.__class__, DeclarativeMeta):
        data = sqlalchemy_to_dict(data)

    response = {
        "message": message,
        "data": data,
        "status": status
    }
    return response

def message_error(
        error: CustomError, 
        message: str
    ) -> Dict:
    response = {
        "status": error.status if hasattr(error, 'status') else 500,
        "message": str(error) if message else "Đã xảy ra lỗi",
        "internal": error.internal_message if hasattr(error, 'internal_message') else str(error)
    }
    return response

def ai_message_error(error: AIError) -> Dict:
    response = {
        "status": error.status if hasattr(error, 'status') else 500,
        "message": str(error) if hasattr(error, 'message') else "Đã xảy ra lỗi",
        "internal": error.internal_message if hasattr(error, 'internal_message') else str(error),
        "ai_raw_response": error.ai_raw_response if hasattr(error, 'ai_raw_response') else None
    }
    return response