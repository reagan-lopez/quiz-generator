import streamlit as st
from store import AppStore
from core import AppCore
from dotenv import load_dotenv
import streamlit as st


store = AppStore()
core = AppCore(store)


# Display the assessment. If the "check_answers" has been clicked, display the answers too.
def display_assessment(assessment):
    for i, question in enumerate(assessment.questions):
        st.write("\n\n")
        st.write(f"{i + 1}) {question.desc}")
        if core.is_multiple_choice(question):  # multiple choice question
            st.write("\n")
            user_selection = {}
            for option in question.options:
                user_selection[option.desc] = st.checkbox(label=option.desc)
            if st.session_state["check_answers"]:
                if core.is_mc_correct(question.options, user_selection):
                    st.success("Correct Answer.")
                else:
                    correct_answers = ""
                    for option in question.options:
                        if option.answer:
                            correct_answers = correct_answers + f"'{option.desc}'" + ","
                    correct_answers = correct_answers.rstrip(",")
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
                if core.is_sc_correct(question.options, user_selection):
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


def main():
    st.set_page_config("Quiz Generator")
    st.title("Generate Quiz from PDFs")

    with st.sidebar:
        st.header("Menu")

        pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Generate Quiz"):
            with st.spinner("Processing.."):
                st.session_state["check_answers"] = False
                if pdfs:
                    model_response = core.generate_assessment(pdfs)
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
