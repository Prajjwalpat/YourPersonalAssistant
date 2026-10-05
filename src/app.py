import streamlit as st
from document_utils import process_documents, SUPPORTED_EXTENSIONS
from conversation import handle_question, render_chat_history
from htmlTemplates import css
from prompt_refiner import refine_prompt_with_llm

def main():
    st.set_page_config(page_title="Chat with Multiple Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "enable_refinement" not in st.session_state:
        st.session_state.enable_refinement = False
    if "ingestion_report" not in st.session_state:
        st.session_state.ingestion_report = []
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    st.header("Chat with Your Documents :books:")

    # Sidebar
    with st.sidebar:
        
        # Add toggle for prompt refinement
        st.markdown("## :gear: Settings")
        st.session_state.enable_refinement = st.toggle("Enable Prompt Refinement", value=st.session_state.enable_refinement)

        st.markdown("## :information_source: About")
        st.info("**Upload your files and press 'Process'** to prepare the documents for questioning.")

        with st.expander("Need Help?"):
            st.write(""" 
            - **Upload files**: Use the file uploader to upload your documents.
            - **Supported formats**: PDF, TXT, CSV, DOCX, Markdown and text-based code files.
            - **Sources**: Expand an answer's sources to inspect filenames, PDF pages and excerpts.
            - **Limitations**: Scanned PDFs need OCR. CSV is text search, not spreadsheet calculations.
            - **Process files**: Click 'Process' to analyze the uploaded documents.
            - **Prompt Refinement**: Enable this feature in Settings to improve question accuracy:
                - When enabled: Your questions will be automatically refined for better context and clarity
                - When disabled: Your questions will be processed exactly as written
                - Recommended: Enable for complex questions or when you need more accurate answers
            """)
        
        # Document uploader
        docs = st.file_uploader("Upload your files here", accept_multiple_files=True, type=SUPPORTED_EXTENSIONS)
        if docs:
            st.session_state.docs = docs
            st.subheader("Uploaded Files")
            for i, doc in enumerate(docs):
                st.write(f"{i+1}. {doc.name}")

        # Processing logic
        if st.button("Process"):
            if docs:
                with st.spinner("Processing documents, please wait..."):
                    st.session_state.chat_history = []
                    st.session_state.conversation = None
                    progress = st.progress(0, text="Reading files...")
                    def update_progress(done, total, name):
                        progress.progress(done / total, text=f"Read {done}/{total}: {name}")
                    st.session_state.processed = process_documents(docs, progress=update_progress)
                    progress.empty()
            else:
                st.error("Please upload at least one document before processing.")

        if st.session_state.ingestion_report:
            st.subheader("Processing report")
            st.dataframe(st.session_state.ingestion_report, hide_index=True, use_container_width=True)
            failed = sum(row["status"] == "Failed" for row in st.session_state.ingestion_report)
            if failed:
                st.warning(f"{failed} file(s) could not be read. See the report for details.")

    # Main chat area
    if st.session_state.processed:
        ready = sum(row["status"] == "Ready" for row in st.session_state.ingestion_report)
        st.success(f"{ready} file(s) indexed into {st.session_state.chunk_count} chunks. You can now ask questions.")

    # Clear before rendering, so no stale messages survive this rerun.
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

    # Use st.chat_input instead of st.text_input
    question = st.chat_input("Ask a question from your document:", disabled=not st.session_state.processed)
    
    if question:
        with st.spinner("Searching your documents and preparing an answer..."):
            refined = refine_prompt_with_llm(question) if st.session_state.enable_refinement else question
            handle_question(question, retrieval_question=refined)

    render_chat_history()

if __name__ == '__main__':
    main()
