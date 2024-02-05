import os
from dotenv import load_dotenv
import logging
import shutil
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import List

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store_name = "vector_store"

logging.basicConfig(filename=f"app.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class Option(BaseModel):
    desc: str
    answer: bool


class Question(BaseModel):
    desc: str
    options: List[Option]


class Assessment(BaseModel):
    questions: List[Question]


# Extract text from pdf.
def get_pdf_text(pdfs):
    pdf_text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text


# Split text into chunks.
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(pdf_text)
    return text_chunks


# Delete vector store
def del_vector_store():
    try:
        shutil.rmtree(vector_store_name)
        logger.info(f"The directory '{vector_store_name}' has been deleted.")
    except Exception as e:
        logger.exception(f"Error deleting the directory: {e}")


# Embed the chunks and insert into vector store.
def set_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(vector_store_name)


# Generate response using the vector store and model.
def get_response(role):
    # Set up a parser + inject instructions into the prompt template.
    pydantic_parser = PydanticOutputParser(pydantic_object=Assessment)
    format_instructions = pydantic_parser.get_format_instructions()

    query = f"""
    Provide a detailed explanation of each of the uploaded documents.
    Make sure that the explanation has the necessary information to test the knowledge of a {role}.
    """
    try:
        vector_store = FAISS.load_local(vector_store_name, embeddings=embeddings)
    except Exception as e:
        logger.exception(f"Error loading vector store:\n{e}")
        return None
    else:
        context_vectors = vector_store.similarity_search(query)

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt_template = """
    You are an expert quiz maker. Based on the Context below, generate a quiz of 5 questions.
    Provide 3 options of answers for each question. One or more of the options should be correct.
    Please ensure that the response is generated as per the Format Instructions below.\n

    Context: \n{context}?\n
    Question: \n{question}\n
    Format Instructions: \n{format_instructions}\n
    """

    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    messages = prompt.format_messages(
        context=context_vectors, question=query, format_instructions=format_instructions
    )

    try:
        output = model(messages=messages)
        logger.info(f"Model output:\n{output}")
    except Exception as e:
        logger.exception(f"Error fetching model output:\n{e}")
        return None

    try:
        response = pydantic_parser.parse(output.content)
    except Exception as e:
        logger.exception(f"Error parsing model output:\n{e}")
        return None
    return response


# Return True if its a multiple choice question.
def is_multiple_choice(question):
    count = 0
    for option in question.options:
        if option.answer:
            count += 1
            if count > 1:
                return True
    return False


# Check the answer of a multiple choice question.
def is_mc_correct(options, user_selection):
    if not user_selection:
        return False
    for option in options:
        if user_selection[option.desc] != option.answer:
            return False
    return True


# Check the answer of a single choice question.
def is_sc_correct(options, user_selection):
    if not user_selection:
        return False
    for option in options:
        if option.desc == user_selection and option.answer is False:
            return False
    return True


# Display the assessment.
# If the "check_answers" has been clicked, display the answers too.
def display_assessment(assessment):
    for i, question in enumerate(assessment.questions):
        numbered_question = f"{i + 1}) {question.desc}"
        st.write("\n\n")
        st.write(numbered_question)
        if is_multiple_choice(question):  # multiple choice question
            st.write("\n")
            user_selection = {}
            for option in question.options:
                user_selection[option.desc] = st.checkbox(label=option.desc)
            if st.session_state["check_answers"]:
                if is_mc_correct(question.options, user_selection):
                    st.success("Correct Answer.")
                else:
                    correct_answers = ""
                    for option in question.options:
                        if option.answer:
                            correct_answers = (
                                correct_answers + f"'{option.desc}'" + " and"
                            )
                    correct_answers = correct_answers.rstrip("  and")
                    st.error(f"Wrong Answer. The correct answer is: {correct_answers}")
        else:  # single choice question
            options = []
            for option in question.options:
                options.append(option.desc)
            user_selection = st.radio(
                label="placeholder",
                label_visibility="hidden",
                options=options,
                index=None,
            )
            if st.session_state["check_answers"]:
                if is_sc_correct(question.options, user_selection):
                    st.success("Correct Answer.")
                else:
                    correct_answer = ""
                    for option in question.options:
                        if option.answer:
                            correct_answer = option.desc
                            break
                    st.error(f"Wrong Answer. The correct answer is: '{correct_answer}'")


# Set the "check_answers" session state when the Check Answers button is clicked.
def handle_check_answers():
    st.session_state["check_answers"] = True


# Streamlit app
def main():
    st.set_page_config("Quiz Generator")
    st.header("Generate Quiz from PDFs")

    # Delete vector store on the first run
    if "first_run" not in st.session_state:
        if os.path.exists(vector_store_name):
            del_vector_store()
        st.session_state["first_run"] = True

    with st.sidebar:
        st.title("Menu")
        pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Generate Quiz"):
            with st.spinner("Processing.."):
                if pdfs:
                    pdf_text = get_pdf_text(pdfs)
                    text_chunks = get_text_chunks(pdf_text)
                    set_vector_store(text_chunks)
                    model_response = get_response(role="Senior Software Engineer")
                    st.session_state["check_answers"] = False

                    if model_response:
                        st.session_state["model_response"] = model_response
                    else:
                        st.warning("Something went wrong. Let's try generating again.")
                else:
                    st.warning("Please select PDFs to upload.")

    if "model_response" in st.session_state:
        display_assessment(st.session_state["model_response"])

        if not st.session_state["check_answers"]:
            st.write("\n\n")
            st.button("Check Answers", on_click=handle_check_answers)

    else:
        st.write("👈 Get started by uploading the PDFs and generating the quiz.")


if __name__ == "__main__":
    main()
