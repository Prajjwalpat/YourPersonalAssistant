import os
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def refine_prompt_with_llm(prompt):
    try:
        # Initialize the LLM client with additional parameters for refinement
        llm = ChatOllama(
            temperature=0.1,  # Low temperature for more deterministic outputs
            num_ctx=4096,
            top_p=0.9,        # Controls diversity of output; set to 0.9 for focused results
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
        )

        # Define the prompt template
        template = ChatPromptTemplate.from_messages([
            ("system", 
            "PROMPT REFINEMENT INSTRUCTIONS:\n\n"
            "You are an AI tool exclusively for refining text prompts. Your task is to make prompts **clearer, concise, and precise** while preserving their **original meaning**. Follow these rules strictly:\n\n"
            "1. Refine prompts without interpreting or answering them.\n"
            "2. Preserve the exact meaning; do not add, remove, or alter intent.\n"
            "3. Focus on fixing ambiguities, redundancies, or dictation errors.\n"
            "4. Maintain the original tone, style, and technical terminology.\n"
            "5. Provide the input unchanged if it is already clear and concise.\n"
            "6. Avoid conversational phrasing, commentary, or explanations.\n\n"
            "Output only the refined prompt text, strictly adhering to these rules."),
            ("human", "{original_prompt}")
        ])

        # Use the LLM to refine the prompt
        refined_prompt = llm.invoke(template.format_messages(original_prompt=prompt)).content.strip()

        # Return the refined prompt
        return refined_prompt

    except Exception as e:
        st.error(f"Error refining prompt: {str(e)}")
        return prompt
