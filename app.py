# app.py

import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI
from document_searcher import DocumentSearcher
from config import Config


class QASystem:
    def __init__(self):
    
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.searcher = DocumentSearcher(Config)
    

    def get_chat_response(self, query: str, context: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Get response from GPT-3.5 Turbo using context
        Args:
            query: User's question
            context: List of relevant text chunks with metadata
        Returns:
            Tuple of (response text, context used)
        """
        # Prepare context string
        context_str = "\n\n".join([
            f"Source: {result['filename']}, Page {result['page_number']}\n" +
            f"{result['page_header']}\n" +
            f"\"{result['text']}\""
            for result in context
        ])

        messages = [
            {
                "role": "system", 
                "content": """You are a knowledgeable spiritual assistant specialized in 
The Mother's works from Sri Aurobindo Ashram, Pondicherry.

Your response must ALWAYS follow this EXACT format:

Source: [filename], Page [page_number]
[page_header]
"[text]"

Where:
- filename: use the exact filename from context
- page_number: use the exact page number from context
- page_header: use the exact page header from context
- text: use the exact text from context

Rules:
1. Start DIRECTLY with the source citation - no introduction or explanation
2. Present quotes exactly as they appear in the original text
3. Maintain original paragraph structure and formatting
4. If using multiple quotes, separate them with blank lines
5. Do not add any additional commentary or text

EXAMPLE OF CORRECT FORMAT:

Source: Words of The Mother-I, Page 123
Words of the Mother – I
"[Exact quote from the source document]"

Remember: Only use content where "The Mother" refers to Mirra Alfassa of Sri Aurobindo Ashram
"""
            },
            {
                "role": "user",
                "content": f"""Context:
                {context_str}

                Question: {query}"""
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content, context
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            raise


    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            return self.searcher.get_embedding(text)
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            raise
    

    def search_similar_chunks(self, query: str) -> List[Dict]:
        """Search for similar text chunks"""
        try:
            return self.searcher.search(
                query=query,
                limit=Config.SEARCH_LIMIT,
                score_threshold=Config.SIMILARITY_THRESHOLD
            )
        except Exception as e:
            st.error(f"Error searching database: {str(e)}")
            raise


def initialize_session_state():
    """Initialize session state variables"""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()


def display_results(results, response=None):
    """
    Display search results in a formatted way
    Args:
        results: List of search results
        response: Optional AI response to display
    """

    if not results:  # Handle case where all results were filtered out
        st.info("No valid results found after filtering. Try rephrasing your question.")
        return

    if response:
        st.markdown("## AI Response")
        print("#### \n\nDebugging: response from llm : ", response)
        
        # Group results by document and page for primary citation
        primary_result = results[0]


        st.markdown(f"**Source: {primary_result['filename']}, Page {primary_result['page_number']}**")

        if primary_result.get('page_header'):
            st.markdown(f"**{primary_result['page_header']}**")

        st.markdown(f"\"...{primary_result['text']}...\"")

    if len(results) <= 1:  # No additional results to show
        return

    st.markdown("### Additional Relevant Passages")
    
    # Group results by document
    grouped_results = {}
    for result in results[1:]:  # Skip first result as it's shown above
        doc_key = result['filename']
        if doc_key not in grouped_results:
            grouped_results[doc_key] = []
        grouped_results[doc_key].append(result)

    # Display grouped results
    for doc_name, doc_results in grouped_results.items():
        # Sort results by page number
        doc_results.sort(key=lambda x: x['page_number'])
        
        # Group consecutive pages
        current_group = []
        current_pages = []
        last_page = None
        
        for result in doc_results:
            page_num = result['page_number']
            
            if last_page is not None and page_num != last_page + 1:
                if current_group:
                    pages_str = f"Page{' ' if len(current_pages) == 1 else 's '}{', '.join(map(str, current_pages))}"

                    title = f"__Source: {doc_name}, {pages_str}__"

                    if current_group[0].get('page_header'):
                        title += f" [▾ {current_group[0]['score']:.0%} relevance]"
                    else:
                        title += f" [▾ {current_group[0]['score']:.0%} relevance]"

                    with st.expander(title, expanded=False):
                        for text in current_group:
                            if text.get('page_header'):
                                st.markdown(f"__{text['page_header']}__")
                                st.markdown(f"\"...{text['text']}...\"")

                current_group = []
                current_pages = []
            
            current_group.append(result)
            current_pages.append(page_num)
            last_page = page_num
        

        if current_group:
            pages_str = f"Page{' ' if len(current_pages) == 1 else 's '}{', '.join(map(str, current_pages))}"
    
            # Format title using Source: format
            title = f"__Source: {doc_name}, {pages_str}__"
            if current_group[0].get('page_header'):
                title += f" [▾ {current_group[0]['score']:.0%} relevance]"
            else:
                title += f" [▾ {current_group[0]['score']:.0%} relevance]"
    
        with st.expander(title, expanded=False):
            for text in current_group:
                if text.get('page_header'):
                    st.markdown(f"__{text['page_header']}__")
                st.markdown(f"\"...{text['text']}...\"")


def main():
    # Page configuration
    st.set_page_config(
        page_title="QnA bot on the Collected Works of The Mother",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🔍 From the Collected Works of The Mother")
    st.markdown("""
    Ask questions about The Mother's works and get AI-powered answers with relevant passages.
    """)
    
    # System status check
    try:
        collection_info = st.session_state.qa_system.searcher.get_collection_info()
        vectors_count = collection_info['vectors_count']
        
        if vectors_count == 0:
            st.warning("⚠️ No documents have been processed yet. Please process documents first.")
            return
        
        # st.success(f"📚 System ready with {vectors_count:,} indexed text passages")
        
    except Exception as e:
        st.error(f"❌ System Error: Could not connect to the database. {str(e)}")
        return
    
    # Search interface
    with st.form("search_form"):
        user_query = st.text_input(
            "Your Question:",
            placeholder="Enter your question about The Mother's works...",
            help="Type your question and press Enter or click 'Search'"
        )
        
        cols = st.columns([1, 4])
        with cols[0]:
            search_button = st.form_submit_button("🔍 Search")
        
    # Process search
    if search_button and user_query:
        if len(user_query.strip()) < 3:
            st.warning("⚠️ Please enter a longer question")
            return
            
        with st.spinner("🔍 Searching and generating response..."):
            try:
                results = st.session_state.qa_system.search_similar_chunks(user_query)
                
                if results:
                    response, context = st.session_state.qa_system.get_chat_response(user_query, results)
                    display_results(results, response)
                else:
                    st.info("ℹ️ No relevant passages found. Try rephrasing your question.")
                    
            except Exception as e:
                st.error(f"❌ Search Error: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")


if __name__ == "__main__":
    main()