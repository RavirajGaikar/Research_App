import json
import streamlit as st
from fpdf import FPDF
import unicodedata
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ArxivRetriever

#This function is used to parse (load) a JSON string and convert it into a Python dictionary (or list, depending on the JSON structure).
def parse_json(raw_output: str):
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON. Raw output: {raw_output}")
        return []
    
#Flatten a list of lists into a single string with double newlines.
def flatten_list(nested_list):
    return "\n\n".join("\n\n".join(sublist) for sublist in nested_list)

#Function to generate PDF 
def generate_pdf(report_text: str) -> bytes:
    if not report_text:
        raise ValueError("No report content provided.")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
        content = report_text
    except Exception:
        pdf.set_font("Helvetica", size=12)
        content = unicodedata.normalize('NFKD', report_text).encode('ascii', 'ignore').decode('ascii')
    
    pdf.multi_cell(0, 10, content)
    pdf_output = pdf.output(dest="S")
    return pdf_output.encode("latin1", errors="replace") if isinstance(pdf_output, str) else bytes(pdf_output)

#Initialize Google Generative AI model.
def initialize_llm(api_key: str):
    return GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.8)

#Initialize the Arxiv retriever.
def initialize_retriever():
    return ArxivRetriever()

#Generate a prompt template for summarizing Arxiv documents.
def get_summary_prompt():
    template = """{doc}\n\n-----------\nUsing the above text, answer in short:\n\n> {question}\n-----------\nIf the question cannot be answered, summarize all factual information, numbers, and statistics."""
    return ChatPromptTemplate.from_template(template)

#Build a summarization chain.
def create_summary_chain(llm):
    return RunnablePassthrough.assign(summary=get_summary_prompt() | llm | StrOutputParser()) | (
        lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}"
    )

#Build a chain for retrieving and summarizing Arxiv documents.
def create_web_search_chain(retriever, llm):
    summary_chain = create_summary_chain(llm)
    return RunnablePassthrough.assign(
        docs=lambda x: retriever.get_summaries_as_docs(x["question"])[:10]
    ) | (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | summary_chain.map()

#Generate search queries for research.
def create_search_query_chain(llm):
    search_prompt = ChatPromptTemplate.from_messages([
        ("user", "Generate 3 Google search queries to find objective information on: {question}\n"
         "Respond strictly as a JSON list: ['query 1', 'query 2', 'query 3']")
    ])
    return search_prompt | llm | StrOutputParser() | (lambda x: parse_json(x))

#Combine search query and web search chains into a complete research pipeline.
def create_full_research_chain(llm, retriever):
    search_chain = create_search_query_chain(llm)
    web_chain = create_web_search_chain(retriever, llm)
    return search_chain | (lambda x: [{"question": q} for q in x]) | web_chain.map()

#Extract research paper titles and URLs from Arxiv documents.
def create_url_list_chain(retriever):
    return RunnablePassthrough.assign(
        url_list=lambda x: [
            {"url": u.metadata['Entry ID'], "Title": u.metadata['Title']}
            for u in retriever.get_summaries_as_docs(x["question"])[:10]
        ]
    ) | (lambda x: {
        "url_list": "\n".join([f"[{item['Title']}]({item['url']})" for item in x["url_list"]])
    })

#Collapse multiple research summaries into a single string.
def create_research_summary_chain(full_research_chain):
    return RunnablePassthrough.assign(research_summary=full_research_chain | flatten_list)

#Build the final research report generation chain.
def create_final_chain(llm, retriever):
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic writer. Generate a high-quality research paper based on input."),
        ("user", """
        Information:\n--------\n{research_summary}\n--------\n
        Write a detailed research paper on: "{question}".\n
        - Include an in-depth literature review, findings, statistics, and APA citations.\n        - Ensure at least 1,200 words.\n        - List references with clickable links from {url_list}.""")
    ])
    full_research_chain = create_full_research_chain(llm, retriever)
    research_summary_chain = create_research_summary_chain(full_research_chain)
    url_list_chain = create_url_list_chain(retriever)
    
    return (
        {"question": RunnablePassthrough(), "research_summary": research_summary_chain, "url_list": url_list_chain}
        | writer_prompt | llm | StrOutputParser()
    )


def main():
    st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")
    st.title("📚 AI Research Assistant")
    
    st.markdown("""
        ### Welcome!
        This tool generates comprehensive research papers by aggregating Arxiv research articles.
        1. Enter your Google API key.
        2. Provide a research topic.
        3. Click *Generate Report*.
        4. Download the generated report as a PDF.
    """)
    
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
    research_topic = st.text_input("Enter Research Topic:")
    
    if st.button("Generate Report"):
        if not google_api_key:
            st.error("API Key is required.")
            return
        if not research_topic:
            st.error("Research topic is required.")
            return
        
        with st.spinner("Generating report..."):
            llm = initialize_llm(google_api_key)
            retriever = initialize_retriever()
            report = create_final_chain(llm, retriever).invoke({"question": research_topic})
        
        st.success("Report generated!")
        st.markdown(report)
        pdf_bytes = generate_pdf(report)
        st.download_button("Download PDF", pdf_bytes, "research_report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
