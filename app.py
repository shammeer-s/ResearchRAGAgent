import streamlit as st
from rag import load_and_embed_code
from agents import (
    run_router_agent, run_search_agent, run_rag_agent,
    run_critic_agent, run_synthesis_agent,
    run_feedback_agent, run_qa_agent
)

# --- Page Configuration ---
st.set_page_config(
    page_title="🔬 Multi-Agent RAG Researcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Function ---
def format_context_and_sources(search_results: list[dict]) -> tuple[str, dict]:
    """Prepares context string and source map for the synthesizer."""
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

# --- Session State Initialization ---
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'source_map' not in st.session_state:
    st.session_state.source_map = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# --- Sidebar: Code RAG Setup ---
with st.sidebar:
    st.title("Code RAG Setup")
    st.markdown("Embed your local Python codebase for the agent to reference.")

    code_dir = st.text_input("Path to your code directory", "./")

    if st.button("Load & Embed Code"):
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

    if st.session_state.retriever:
        st.success("Code RAG is **Active**")

# --- Main App Interface ---
st.title("🔬 Multi-Agent RAG Researcher")
st.markdown("This app uses a team of AI agents to research topics, reference your code, and generate a cited report.")

query = st.text_input("Enter your research query:",
                      placeholder="e.g., Explain the 'Attention Is All You Need' paper and how it relates to my 'transformer.py' file")

if st.button("Run Research", type="primary"):
    if not query:
        st.error("Please enter a query.")
    else:
        # Reset session
        st.session_state.report = None
        st.session_state.source_map = None
        st.session_state.qa_history = []

        # This status box shows the agent's "thoughts"
        with st.status("Agents at work...", expanded=True) as status:
            try:
                # 1. Router Agent
                status.write("🧠 Running Router Agent to classify query...")
                decision = run_router_agent(query)
                status.write(f"✅ Router decision: **{decision}**")

                # 2. Run Search + RAG in parallel
                status.write("📚 Running Search Agent...")
                search_results = run_search_agent(query, decision)

                code_context = ""
                if st.session_state.retriever:
                    status.write("💻 Running Code RAG Agent...")
                    code_context = run_rag_agent(query, st.session_state.retriever)

                # 3. Critic Agent
                status.write(f"🧐 Running Critic Agent to review {len(search_results)} snippets...")
                filtered_results, critic_reasoning = run_critic_agent(query, search_results)
                status.write(f"✅ Critic decision: {critic_reasoning}")

                # 4. Format Context
                research_context, source_map = format_context_and_sources(filtered_results)

                # 5. Synthesis Agent
                status.write("✍️ Running Synthesis Agent to write report...")
                report = run_synthesis_agent(query, research_context, code_context)

                # 6. Feedback Agent (Agent Loop)
                status.write("🕵️ Running Feedback Agent for quality check...")
                feedback = run_feedback_agent(query, report)

                if "no." in feedback.lower():
                    status.warning(f"Feedback Agent triggered loop: \"{feedback}\"")
                    status.write("♻️ Re-running Synthesis Agent with feedback...")
                    report = run_synthesis_agent(query, research_context, code_context, feedback=feedback)

                st.session_state.report = report
                st.session_state.source_map = source_map
                status.update(label="Research Complete!", state="complete")

            except Exception as e:
                status.update(label=f"An error occurred: {e}", state="error")


# --- Display Report ---
if st.session_state.report:
    st.markdown("---")
    st.markdown("### Generated Report")
    st.info(st.session_state.report)

    with st.expander("Show Sources"):
        if st.session_state.source_map:
            st.json(st.session_state.source_map)
        else:
            st.write("No sources were cited.")

    # --- QA Bot Section ---
    st.markdown("---")
    st.markdown("### 💬 Explainable QA Bot")
    st.markdown("Ask a follow-up question about the report.")

    # Display QA history
    for q, a in st.session_state.qa_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    qa_query = st.chat_input("Your question...")

    if qa_query:
        st.chat_message("user").write(qa_query)
        with st.spinner("Thinking..."):
            answer = run_qa_agent(st.session_state.report, qa_query)
            st.session_state.qa_history.append((qa_query, answer))
            st.chat_message("assistant").write(answer)