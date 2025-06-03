import streamlit as st
# LangChain imports for the Study Buddy section
from langchain_google_genai import GoogleGenerativeAI as LangChainGoogleGenerativeAI # Alias for clarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Still useful for text-based PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA # Not directly used for chat in this version
from langchain.prompts import PromptTemplate
# Standard Python imports
import os
import tempfile
import hashlib
import time

# --- OCR Specific Imports (using Gemini directly) ---
import google.generativeai as genai # This is the primary SDK for Gemini

# --- App Configuration & Title ---
st.set_page_config(page_title="Gemini Study Buddy Pro (Features++)", layout="wide")
st.title("📚 Gemini Study Buddy Pro (Features++)")

# --- API Key Configuration ---
try:
    GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY_GEMINI", os.getenv("GOOGLE_API_KEY_GEMINI"))
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_GEMINI")

if not GEMINI_API_KEY:
    st.error("🔴 Gemini API Key (GOOGLE_API_KEY_GEMINI) not found. Please set it. All features will be disabled.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Initialize LLM and Embeddings ---
llm_studybuddy = None # For summaries, flashcards
llm_qna = None      # For chat Q&A
embeddings_studybuddy = None
try:
    llm_studybuddy = LangChainGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.5, google_api_key=GEMINI_API_KEY)
    llm_qna = LangChainGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.7, google_api_key=GEMINI_API_KEY) # Using your specified model
    embeddings_studybuddy = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document", google_api_key=GEMINI_API_KEY)
except Exception as e:
    st.sidebar.error(f"Error initializing Gemini models: {e}")

# --- Session State Management ---
if 'ocr_text_output' not in st.session_state:
    st.session_state.ocr_text_output = None
if 'ocr_file_name' not in st.session_state:
    st.session_state.ocr_file_name = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_file_hash' not in st.session_state:
    st.session_state.processed_file_hash = None
if 'documents_for_direct_use' not in st.session_state:
    st.session_state.documents_for_direct_use = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_doc_chat_hash' not in st.session_state:
    st.session_state.current_doc_chat_hash = None
if 'last_used_sources' not in st.session_state: # For Q&A source highlighting
    st.session_state.last_used_sources = []


# =============================================
# SECTION 1: OCR PDF (Using Gemini Multimodal)
# =============================================
# ... (OCR section remains the same as your last provided version) ...
st.sidebar.markdown("---")
st.sidebar.header("📄 OCR Scanned PDF (with Gemini)")
ocr_uploaded_file = st.sidebar.file_uploader("Upload a scanned PDF for Gemini OCR", type="pdf", key="gemini_ocr_uploader")

def perform_ocr_with_gemini(pdf_file_uploader_object):
    try:
        st.sidebar.write("Uploading PDF to Gemini File API...")
        uploaded_gemini_file = genai.upload_file(
            path=pdf_file_uploader_object,
            display_name=pdf_file_uploader_object.name,
            mime_type=pdf_file_uploader_object.type
        )
        st.sidebar.write(f"File '{uploaded_gemini_file.display_name}' uploaded. URI: {uploaded_gemini_file.uri}. Mime Type: {pdf_file_uploader_object.type}")
        st.sidebar.write("Extracting text with Gemini 1.5 Flash...")
        model_ocr = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-04-17")
        prompt = [
            "Please perform OCR on the provided PDF document and extract all text content.",
            "Present the extracted text clearly. If there are multiple pages, try to indicate page breaks with something like '--- Page X ---' if possible, or just provide the continuous text.",
            "Focus solely on extracting the text as accurately as possible from the document.",
            uploaded_gemini_file 
        ]
        response = model_ocr.generate_content(prompt, request_options={"timeout": 600})
        try:
            genai.delete_file(uploaded_gemini_file.name)
            st.sidebar.write(f"Temporary file '{uploaded_gemini_file.display_name}' deleted from Gemini File API.")
        except Exception as e_delete:
            st.sidebar.warning(f"Could not delete temporary file from Gemini File API: {e_delete}")
        return response.text
    except Exception as e:
        st.sidebar.error(f"Gemini OCR Error: {e}")
        if 'uploaded_gemini_file' in locals() and hasattr(uploaded_gemini_file, 'name'):
            try: genai.delete_file(uploaded_gemini_file.name)
            except: pass
        return None

if ocr_uploaded_file is not None:
    if st.sidebar.button("✨ Perform Gemini OCR", key="gemini_ocr_button"):
        st.session_state.ocr_text_output = None 
        st.session_state.ocr_file_name = None
        with st.spinner("Performing OCR with Gemini... This may take a while for large files."):
            extracted_text = perform_ocr_with_gemini(ocr_uploaded_file)
            if extracted_text:
                st.session_state.ocr_text_output = extracted_text
                st.session_state.ocr_file_name = f"gemini_ocr_output_{os.path.splitext(ocr_uploaded_file.name)[0]}.txt"
                st.sidebar.success("Gemini OCR Complete!")
            else:
                st.sidebar.error("Gemini OCR failed or no text was extracted.")

if st.session_state.ocr_text_output:
    st.sidebar.subheader("Gemini OCR Result:")
    st.sidebar.download_button(
        label="📥 Download OCR'd Text",
        data=st.session_state.ocr_text_output.encode('utf-8'),
        file_name=st.session_state.ocr_file_name,
        mime="text/plain",
        key="download_gemini_ocr"
    )
    with st.sidebar.expander("Preview Gemini OCR Text (First 1000 Chars)"):
        st.text(st.session_state.ocr_text_output[:1000] + "...")


# =============================================
# SECTION 2: Study Buddy Q&A and Tools
# =============================================
st.sidebar.markdown("---")
st.sidebar.header("🧠 Study Buddy Tools")
study_uploaded_file = st.sidebar.file_uploader(
    "Upload TEXT-READABLE PDF or TXT for Q&A, Summary, etc.", 
    type=["pdf", "txt"], 
    key="study_uploader",
    help="If your PDF is scanned, please use the 'OCR Scanned PDF' section above first and then upload the downloaded .txt file here."
)

if study_uploaded_file is not None and GEMINI_API_KEY and llm_studybuddy and embeddings_studybuddy:
    file_bytes = study_uploaded_file.getvalue()
    current_file_hash = hashlib.md5(file_bytes).hexdigest()

    if current_file_hash != st.session_state.processed_file_hash:
        st.sidebar.info(f"New file '{study_uploaded_file.name}' for Study Buddy. Processing...")
        st.session_state.vector_store = None
        st.session_state.documents_for_direct_use = None
        st.session_state.chat_history = [] 
        st.session_state.current_doc_chat_hash = current_file_hash 
        st.session_state.last_used_sources = [] # Clear sources for new doc
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{study_uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            if study_uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
            if study_uploaded_file.type == "application/pdf" and (not documents or not any(doc.page_content.strip() for doc in documents)):
                st.sidebar.error("Uploaded PDF for Study Buddy has no extractable text. Use OCR section first for scanned PDFs.")
                os.remove(tmp_file_path)
                st.session_state.processed_file_hash = None
            else:
                st.session_state.documents_for_direct_use = documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                texts = text_splitter.split_documents(documents)
                valid_texts = [text for text in texts if text.page_content and text.page_content.strip()]
                if not valid_texts:
                    st.sidebar.error("No valid text chunks after splitting for Study Buddy.")
                else:
                    with st.spinner("Creating embeddings for Study Buddy..."):
                        st.session_state.vector_store = Chroma.from_documents(documents=valid_texts, embedding=embeddings_studybuddy)
                    st.session_state.processed_file_hash = current_file_hash
                    st.sidebar.success(f"✅ '{study_uploaded_file.name}' ready for Study Buddy!")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception as e:
            st.sidebar.error(f"Error processing Study Buddy file: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
            st.session_state.vector_store = None
            st.session_state.documents_for_direct_use = None
            st.session_state.processed_file_hash = None
            st.session_state.chat_history = []
            st.session_state.current_doc_chat_hash = None
            st.session_state.last_used_sources = []


# --- Main Interaction Area for Study Buddy Tools ---
if st.session_state.get('vector_store') and st.session_state.get('documents_for_direct_use') and GEMINI_API_KEY and llm_qna and llm_studybuddy:
    st.markdown("---")
    if st.session_state.current_doc_chat_hash != st.session_state.processed_file_hash:
        st.session_state.chat_history = []
        st.session_state.current_doc_chat_hash = st.session_state.processed_file_hash
        st.session_state.last_used_sources = []
        
    header_file_name = "your document"
    if study_uploaded_file and hasattr(study_uploaded_file, 'name'):
        if st.session_state.processed_file_hash == hashlib.md5(study_uploaded_file.getvalue()).hexdigest():
            header_file_name = study_uploaded_file.name
            
    st.header(f"🛠️ Study Tools for: {header_file_name}")
    
    query_type_key_suffix = st.session_state.processed_file_hash or "default_study_tools"
    query_type = st.radio(
        "What do you want to do with the text-readable document?",
        ("Chat & Ask Questions", "Generate Flashcards (Term>>Definition)", "Summarize Document"),
        key=f"query_type_{query_type_key_suffix}"
    )

    if query_type == "Chat & Ask Questions":
        st.subheader("💬 Chat with your Document")
        # Display chat history
        for item in st.session_state.chat_history:
            role = item.get("role")
            content = item.get("content")
            sources = item.get("sources") # Get sources if available
            with st.chat_message(role):
                st.markdown(content)
                if role == "ai" and sources: # Display sources for AI messages
                    with st.expander("📚 View Sources Used", expanded=False):
                        for i, source_doc in enumerate(sources):
                            page_label = source_doc.metadata.get('page', 'N/A')
                            st.caption(f"Source {i+1} (Page: {page_label}):")
                            st.markdown(f"> {source_doc.page_content[:300]}...") # Show a snippet
                            st.markdown("---")
        
        user_question = st.chat_input("Ask a follow-up question or a new question...", key=f"chat_input_{query_type_key_suffix}")

        if st.button("Clear Chat History", key=f"clear_chat_{query_type_key_suffix}"):
            st.session_state.chat_history = []
            st.session_state.last_used_sources = []
            st.rerun()

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question, "sources": None})
            with st.chat_message("user"): # Display current user question immediately
                st.markdown(user_question)

            with st.spinner("Thinking..."):
                history_for_prompt_list = [f"Previous {item['role']}: {item['content']}" for item in st.session_state.chat_history[:-1]]
                history_for_prompt = "\n".join(history_for_prompt_list)
                
                retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # Retrieve top 3 chunks
                
                prompt_template_chat_qa = """You are an helpful expert in all fields of study and the best generalist on earth who understands everything well. Use the following pieces of context from a document AND the preceding chat history to answer the user's current question.
                Provide a explanatory and elaborative answer based SOLELY on the provided context and chat history.
                If the question is a follow-up, use the chat history to understand the context of the follow-up.
                If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
                Explain the concepts clearly and show your thinking.

                Chat History (if any):
                {chat_history}

                Retrieved Context from Document:
                {context}

                User's Current Question: {question}
                
                Elaborative Answer:"""
                CHAT_QA_PROMPT = PromptTemplate(
                    template=prompt_template_chat_qa, input_variables=["chat_history", "context", "question"]
                )
                
                try:
                    retrieved_docs = retriever.invoke(user_question) # Get source documents
                    st.session_state.last_used_sources = retrieved_docs # Store for display

                    context_for_prompt = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    full_chat_prompt_str = CHAT_QA_PROMPT.format(
                        chat_history=history_for_prompt if history_for_prompt else "No previous chat history for this question.",
                        context=context_for_prompt,
                        question=user_question
                    )
                    
                    ai_response_text = llm_qna.invoke(full_chat_prompt_str)
                    st.session_state.chat_history.append({"role": "ai", "content": ai_response_text, "sources": retrieved_docs})
                    # Rerun to display the new AI message and its sources
                    st.rerun()

                except Exception as e:
                    error_message = f"Error getting answer from Gemini: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "ai", "content": f"Sorry, an error occurred: {e}", "sources": None})
                    st.rerun() # Rerun to show the error message in chat

    elif query_type == "Generate Flashcards (Term>>Definition)":
        if st.button("Generate Flashcards", key=f"flashcard_button_{query_type_key_suffix}"):
            with st.spinner("Generating flashcards..."):
                all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                context_limit_flashcards = 300000 
                prompt_template_flashcards = f"""
                Based ONLY on the following text, identify key terms and their definitions.
                Format each as 'Term>>Definition'. Each flashcard should be on a new line.
                Text:
                ---
                {all_doc_text[:context_limit_flashcards]}
                ---
                Flashcards:
                """
                try:
                    response_text = llm_studybuddy.invoke(prompt_template_flashcards)
                    st.subheader("Flashcards:")
                    st.text_area("Copy these flashcards:", response_text, height=400, key=f"flashcard_output_{query_type_key_suffix}")
                except Exception as e:
                    st.error(f"Error generating flashcards: {e}")
    
    elif query_type == "Summarize Document":
        summary_session_key = f"summary_text_{query_type_key_suffix}"
        if summary_session_key not in st.session_state:
            st.session_state[summary_session_key] = ""

        summary_length = st.selectbox("Select summary length:", ("Short", "Medium", "Detailed"), key=f"summary_length_{query_type_key_suffix}")
        if st.button("Summarize", key=f"summary_button_{query_type_key_suffix}"):
            st.session_state[summary_session_key] = "" 
            with st.spinner("Summarizing..."):
                if st.session_state.get('documents_for_direct_use'):
                    all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                    context_limit_summary = 500000
                    length_instruction = {
                        "Short": "Provide a very brief, one-paragraph executive summary.",
                        "Medium": "Provide a multi-paragraph summary covering the main sections and key arguments.",
                        "Detailed": "Provide a comprehensive and elaborative summary, breaking down complex topics and highlighting all major sections, arguments, examples, and conclusions found in the text."
                    }
                    prompt_template_summary = f"""
                    Based ONLY on the following text, {length_instruction[summary_length]}
                    Format the output in Markdown.
                    Text:
                    ---
                    {all_doc_text[:context_limit_summary]}
                    ---
                    {summary_length} Summary (Formatted in Markdown):
                    """
                    try:
                        response_text_summary = llm_studybuddy.invoke(prompt_template_summary)
                        st.session_state[summary_session_key] = response_text_summary
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                        st.session_state[summary_session_key] = f"Error generating summary: {e}"
                else:
                    st.warning("No document loaded to summarize.")
                    st.session_state[summary_session_key] = "No document loaded to summarize."
        
        if st.session_state.get(summary_session_key):
            st.subheader(f"{summary_length} Summary:")
            current_summary_text = st.session_state[summary_session_key]
            st.markdown(current_summary_text)
            
            # --- NEW: Export Summary Button ---
            if current_summary_text and "Error generating summary" not in current_summary_text and "No document loaded" not in current_summary_text:
                st.markdown("---")
                summary_file_name = f"summary_{header_file_name.replace(' ', '_').split('.')[0]}_{summary_length.lower()}.md"
                st.download_button(
                    label="📥 Download Summary (Markdown)",
                    data=current_summary_text.encode('utf-8'), # Encode to bytes
                    file_name=summary_file_name,
                    mime="text/markdown",
                    key=f"download_summary_{query_type_key_suffix}"
                )
            # --- End Export Summary Button ---

            # Optional: Text area for manual copy if preferred by user
            # st.markdown("---")
            # st.text_area(
            #     label="Raw Markdown Summary (for copying):",
            #     value=current_summary_text if current_summary_text and "Error" not in current_summary_text and "No document" not in current_summary_text else "Summary not generated or error occurred.",
            #     height=200,
            #     key=f"summary_raw_text_area_{query_type_key_suffix}"
            # )


elif not GEMINI_API_KEY:
    st.warning("Gemini features are disabled as the Gemini API Key is not provided.")
else:
    st.info("👋 Upload a text-readable document in the sidebar to use the Study Buddy tools. For scanned PDFs, use the OCR section first.")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Streamlit, LangChain & Google Gemini")