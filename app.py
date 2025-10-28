import streamlit as st
import re
from rag import load_and_embed_code
from agents import (
    run_router_agent, run_search_agent, run_rag_agent,
    run_critic_agent, run_synthesis_agent,
    run_feedback_agent, run_qa_agent
)

st.set_page_config(
    page_title="RAG Researcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Base */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Main container */
    [data-testid="stAppViewContainer"] {
        background: #ffffff; /* White background */
        color: #1f1f1f; /* Dark text */
    }
    [data-testid="stHeader"] {
        display: none;
    }
    
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8f9fa; /* Light gray */
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] h1 {
        color: #1a73e8; /* Google Blue */
        font-weight: 600;
    }
    
    [data-testid="stTextInput"] input::placeholder {
        color: #1f1f1f; /* Google Blue */
    }
    
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] p {
        color: #333333CC; /* Dark text */
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Title */
    h1, h2, h3 {
        color: #1f1f1f; /* Dark text */
        font-weight: 600;
    }
    
    /* Buttons */
    [data-testid="stButton"] button {
        background: #1a73e8; /* Google Blue */
        color: #ffffff;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    [data-testid="stButton"] button:hover {
        background: #1865c9;
    }
    
    /* Text Input */
    [data-testid="stTextInput"] input {
        background: #f1f3f4; /* Light gray input */
        color: #1f1f1f; /* Dark text */
        border: 1px solid #dadce0;
        border-radius: 0.5rem;
    }
    [data-testid="stTextInput"] label {
        color: #5f6368; /* Gray text */
    }
    
    /* Report Box (st.info) */
    .report-box {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        color: #3c4043; /* Dark text */
        border-radius: 0.5rem;
        padding: 1.5rem;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    [data-testid="stExpander"] summary {
        color: #5f6368;
        font-weight: 500;
    }
    [data-testid="stExpander"] summary:hover {
        color: #1a73e8;
    }
    
    /* --- Gemini-style Chat UI (Light) --- */

    [data-testid="stChatMessage"] {
        background: none;
        border: none;
        padding: 0;
        margin-bottom: 1.5rem;
    }

    [data-testid="stChatAvatar"] {
        width: 2.5rem;
        height: 2.5rem;
        background: #f1f3f4;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageContentAssistant"]) [data-testid="stChatAvatar"] {
        background: #1a73e8;
        background: -webkit-linear-gradient(160deg, #1a73e8 20%, #9333ea 50%, #f2994a 80%);
        background: linear-gradient(160deg, #1a73e8 20%, #9333ea 50%, #f2994a 80%);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageContentAssistant"]) [data-testid="stChatAvatar"]::after {
        content: "";
        font-size: 1.25rem;
        content: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v.01M12 21v.01M3 12h.01M21 12h.01M5.64 5.64l.01.01M18.36 18.36l.01.01M5.64 18.36l.01-.01M18.36 5.64l.01-.01"/></svg>');
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageContentUser"]) [data-testid="stChatAvatar"]::after {
        content: "";
        font-size: 1.25rem;
        content: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%233c4043" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>');
    }
    
    [data-testid="stChatMessageContentUser"] {
        background: #f1f3f4;
        color: #3c4043; /* Dark text */
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
    }
    
    [data-testid="stChatMessageContentAssistant"] {
        background: none;
        color: #3c4043; /* Dark text */
        padding: 0.75rem 0;
    }

    [data-testid="stChatInput"] {
        background: #f1f3f4;
        border: 1px solid #dadce0;
        border-radius: 2rem;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        max-width: 900px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] input {
        background: transparent;
        border: none;
        color: #1f1f1f; /* Dark text */
    }
    [data-testid="stChatInput"] input::placeholder {
        color: #1f1f1f; /* Dark text */
    }
    [data-testid="stChatInput"] button {
        border-radius: 50%;
        background: #1a73e8;
        color: white;
    }
    [data-testid="stChatInput"] button:hover {
        background: #1865c9;
    }

    /* --- Inline Source Citation (Light) --- */
    .source-link {
        display: inline-block;
        padding: 0.1rem 0.4rem;
        margin: 0 0.1rem;
        background-color: #e8f0fe; /* Light blue */
        color: #1967d2; /* Blue */
        border: 1px solid #d2e3fc;
        border-radius: 0.375rem;
        font-size: 0.8rem;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .source-link:hover {
        background-color: #dceafb;
        border-color: #c6dafb;
        color: #185abc;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


def format_context_and_sources(search_results):
    context_str = ""
    source_map = {}

    if not search_results:
        return "No information found.", {}

    for i, res in enumerate(search_results, 1):
        context_str += f"Snippet: {res['snippet']}\n[Source {i}]\n\n"
        source_map[f"[Source {i}]"] = {
            "url": res['source_url'],
            "title": res['title']
        }
    return context_str, source_map

def format_report_with_links(report, source_map):
    if not source_map:
        return report

    def replace_match(match):
        source_key = match.group(0)
        if source_key in source_map:
            source_info = source_map[source_key]
            url = source_info['url']
            title = source_info['title'].replace('"', '&quot;') # Escape quotes for HTML title

            return (
                f' <a href="{url}" target="_blank" class="source-link" title="{title}">'
                f'{source_key.replace("[", "").replace("]", "")}'
                f'</a>'
            )
        return source_key # Return original if not found (should not happen)


    report_with_links = re.sub(r'\[Source \d+\]', replace_match, report)
    return report_with_links


if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'report' not in st.session_state:

    st.session_state.report = None
if 'source_map' not in st.session_state:
    st.session_state.source_map = None

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []


with st.sidebar:
    st.title("Code RAG Setup")
    st.markdown("Embed your local Python codebase for the agent to reference.")

    code_dir = st.text_input("Path to your code directory", "./")

    if st.button("Load & Embed Code", type="primary"):
        if not code_dir:
            st.error("Please provide a directory path.")
        else:
            with st.spinner("Embedding code... This may take a moment."):
                try:
                    retriever = load_and_embed_code(code_dir)
                    st.session_state.retriever = retriever
                    st.success("Codebase embedded successfully!")
                except Exception as e:
                    st.error(f"Error embedding code: {e}")

st.title("Multi-Agent RAG Researcher")
st.markdown("This app uses a team of AI agents to research topics, reference your code, and generate a cited report.")

query = st.text_input("Enter your research query:",
                      placeholder="e.g., Explain 'Attention Is All You Need' and how it relates to my 'transformer.py' file")

if st.button("Run Research", type="primary"):

    if not query:
        st.error("Please enter a query.")
    else:

        st.session_state.report = None
        st.session_state.source_map = None
        st.session_state.qa_history = []


        with st.spinner("Agents at work... please wait..."):
            try:

                # Router Agent
                print("Running Router Agent...")
                decision = run_router_agent(query)

                # Run Search + RAG in parallel
                print("Running Search Agent...")
                search_results = run_search_agent(query, decision)

                code_context = ""

                if st.session_state.retriever:
                    print("Running Code RAG Agent...")
                    code_context = run_rag_agent(query, st.session_state.retriever)
                else:
                    print("No code retriever available. Skipping code context.")

                # Critic Agent
                filtered_results, critic_reasoning = run_critic_agent(query, search_results)
                print(f"Critic decision: {critic_reasoning}")

                # Format Context
                research_context, source_map = format_context_and_sources(filtered_results)

                # Synthesis Agent
                report = run_synthesis_agent(query, research_context, code_context)

                # Feedback Agent (Agent Loop)
                feedback = run_feedback_agent(query, report)

                if "no." in feedback.lower():
                    print(f"Feedback Agent triggered loop: \"{feedback}\"")
                    report = run_synthesis_agent(query, research_context, code_context, feedback=feedback)

                st.session_state.report = report
                st.session_state.source_map = source_map
                
                print("Research Complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                
                print(f"An error occurred: {e}")



if st.session_state.report:
    st.markdown("---")
    
    st.markdown("## Generated Report")

    formatted_report = format_report_with_links(
        
        st.session_state.report,
        st.session_state.source_map
    )


    st.markdown(f'<div class="report-box">{formatted_report}</div>', unsafe_allow_html=True)

    with st.expander("View All Cited Sources (List View)"):
        if st.session_state.source_map:
            
            for key, source in st.session_state.source_map.items():
                st.markdown(f"**{key}:** [{source['title']}]({source['url']})")
        else:
            
            st.write("No sources were cited for this report.")

    st.markdown("---")
    st.markdown("### Explainable QA Bot")
    
    st.markdown("Ask a follow-up question about the report.")

    # Display QA history
    for q, a in st.session_state.qa_history:
        
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            
            st.write(a)

    qa_query = st.chat_input("Your question...")

    if qa_query:
        
        with st.chat_message("user"):
            st.write(qa_query)

        with st.chat_message("assistant"):
            
            with st.spinner("Thinking..."):
                answer = run_qa_agent(st.session_state.report, qa_query)
                st.session_state.qa_history.append((qa_query, answer))
                
                st.write(answer)
