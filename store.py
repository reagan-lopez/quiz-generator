import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import Error
from langchain.vectorstores.pgvector import PGVector
from core import logger

load_dotenv()
CONNECTION_STRING = f'{os.getenv("PG_DBNAME")}://{os.getenv("PG_USER")}:{os.getenv("PG_PASSWORD")}@{os.getenv("PG_HOST")}:{os.getenv("PG_PORT")}/{os.getenv("PG_DBNAME")}'
SUCCESS_CODE = 0
SUCCESS_MESSAGE = "Operation completed successfully."


def open_connection():
    try:
        connection = psycopg2.connect(
            dbname=os.getenv("PG_DBNAME"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
        )
        return connection
    except (Exception, Error) as error:
        logger.error("Error connecting to PostgreSQL:", error)


def close_connection(connection, cursor):
    if connection:
        cursor.close()
        connection.close()
        logger.error("PostgreSQL connection is closed")


def insert_user(username, email, full_name, hashed_password, user_type):
    try:
        connection = open_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO user_info (username, email, full_name, hashed_password, user_type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """
        cursor.execute(query, (username, email, full_name, hashed_password, user_type))

        connection.commit()

        user_id = cursor.fetchone()[0]
        logger.info(f"User created with ID: {user_id}")
        return SUCCESS_CODE, SUCCESS_MESSAGE

    except (Exception, Error) as error:
        logger.error("Error inserting data:", error)
        return error.pgcode, error.pgerror

    finally:
        close_connection(connection, cursor)


def insert_job(user_id, job_role, job_description):
    try:
        connection = open_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO job_info (user_id, job_role, job_description)
            VALUES (%s, %s, %s)
            RETURNING id;
        """
        cursor.execute(query, (user_id, job_role, job_description))

        connection.commit()

        job_id = cursor.fetchone()[0]
        logger.info(f"Job created with ID: {job_id}")
        return SUCCESS_CODE, SUCCESS_MESSAGE

    except (Exception, Error) as error:
        logger.error("Error inserting data:", error)
        return error.pgcode

    finally:
        close_connection(connection, cursor)


def insert_question(job_id, question_text, options, answers):
    try:
        connection = open_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO question (job_id, question, options, answers)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """
        cursor.execute(query, (job_id, question_text, options, answers))

        connection.commit()

        question_id = cursor.fetchone()[0]
        logger.info(f"Question created with ID: {question_id}")
        return SUCCESS_CODE, SUCCESS_MESSAGE

    except (Exception, Error) as error:
        logger.error("Error inserting question:", error)
        return error.pgcode

    finally:
        close_connection(connection, cursor)


def insert_response(user_id, question_id, answers):
    try:
        connection = open_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO response (user_id, question_id, answers)
            VALUES (%s, %s, %s)
            RETURNING id;
        """
        cursor.execute(query, (user_id, question_id, answers))

        connection.commit()

        response_id = cursor.fetchone()[0]
        logger.info(f"Response created with ID: {response_id}")
        return SUCCESS_CODE, SUCCESS_MESSAGE

    except (Exception, Error) as error:
        logger.error("Error inserting response:", error)
        return error.pgcode

    finally:
        close_connection(connection, cursor)


def insert_result(job_id, question_id, result):
    try:
        connection = open_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO result (job_id, question_id, result)
            VALUES (%s, %s, %s)
            RETURNING id;
        """
        cursor.execute(query, (job_id, question_id, result))

        connection.commit()

        result_id = cursor.fetchone()[0]
        logger.info(f"Result created with ID: {result_id}")
        return SUCCESS_CODE, SUCCESS_MESSAGE

    except (Exception, Error) as error:
        logger.error("Error inserting result:", error)
        return error.pgcode

    finally:
        close_connection(connection, cursor)
