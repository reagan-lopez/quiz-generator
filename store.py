from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)


class AppStore:
    def __init__(self):
        self.engine = create_engine(
            "postgresql://admin:adminpassword@localhost:5432/prescreendb"
        )
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_all_users(self):
        return self.session.query(User).all()

    def add_user(self, email, name):
        try:
            new_user = User(email=email, name=name)
            self.session.add(new_user)
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
