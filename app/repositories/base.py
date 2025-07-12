from app.utils import CustomError
from typing import Type, TypeVar, Generic, Union
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import cast, String
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from uuid import UUID

T = TypeVar('T')


class BaseRepository(Generic[T]):
    def __init__(
        self, 
        model: Type[T], 
        session: Session
    ):
        self.model = model
        self.session = session

    def save(
        self, 
        entity: T
    ) -> tuple[T, CustomError]:
        try:
            self.session.add(entity)
            self.session.commit()
            return entity, None
        except SQLAlchemyError as e:
            self.session.rollback()
            return None, e

    def get_by_id(
        self, 
        id: Union[int, UUID]
    ) -> tuple[T, CustomError]:
        try:
            if isinstance(id, UUID):
                result = self.session.query(self.model).filter(
                    cast(self.model.id, String) == str(id)
                ).first()
            else:
                result = self.session.query(self.model).filter_by(id=id).first()
            return result, None
        except SQLAlchemyError as e:
            return None, e
        except Exception as e:
            return None, CustomError(str(e))
        finally:
            self.session.close()
            
    def update_by_id(
        self, 
        id: Union[int, UUID], 
        update_data: dict
    ) -> bool:
        try:
            if isinstance(id, UUID):
                updated = self.session.query(self.model).filter(
                    cast(self.model.id, String) == str(id)
                ).update(update_data)
            else:
                updated = self.session.query(self.model).filter_by(id=id).update(update_data)
            
            self.session.commit()
            return updated > 0
        except SQLAlchemyError as e:
            self.session.rollback()
            return False
        except Exception as e:
            self.session.rollback()
            return False
        
    def get_all(self) -> tuple[list[T], CustomError]:
        try:
            result = self.session.query(self.model).all()
            return result, None
        except SQLAlchemyError as e:
            return None, e
        except Exception as e:
            return None, CustomError(str(e))
        finally:
            self.session.close()
        
    def get_list_by_key(
        self, 
        key:str
    ) -> tuple[list[T], CustomError]:
        try:
            result = self.session.query(self.model).filter_by(key=key).all()
            return result, None
        except SQLAlchemyError as e:
            return None, e
        except Exception as e:
            return None, CustomError(str(e))    
        finally:
            self.session.close()
    
    def get_by_key_and_condition(
        self, 
        key: str, 
        condition
    ) -> tuple[T, CustomError]:
        try:
            result = self.session.query(self.model).filter_by(key=key).filter(condition).first()
            return result, None
        except SQLAlchemyError as e:
            return None, e
        except Exception as e:
            return None, CustomError(str(e))
        finally:
            self.session.close()
    
    def update_by_condition(
        self, 
        condition, 
        update_data
    ):
        try:
            self.session.query(self.model).filter(condition).update(update_data)
            self.session.commit()
            return True, None
        except SQLAlchemyError as e:
            self.session.rollback()
            return False, e
        except Exception as e:
            self.session.rollback()
            return False, CustomError(str(e))
        finally:
            self.session.close()
