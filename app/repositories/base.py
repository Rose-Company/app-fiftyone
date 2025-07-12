from app.utils import CustomError
from typing import Type, TypeVar, Generic
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.exc import SQLAlchemyError

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
        id: int
    ) -> tuple[T, CustomError]:
        try:
            result = self.session.query(self.model).filter_by(id=id).first()
            return result, None
        except SQLAlchemyError as e:
            return None, e
        except Exception as e:
            return None, CustomError(str(e))
        finally:
            self.session.close()
        
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
