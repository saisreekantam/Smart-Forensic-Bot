"""
Simple Web Frontend for Project Sentinel
Demo interface for forensic officers to interact with the case management system
"""

import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List

# Configure Streamlit page
st.set_page_config(
    page_title="Project Sentinel - Forensic Investigation Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE = "http://localhost:8000"

def main():
    """Main application interface"""
    
    st.title("🔍 Project Sentinel")
    st.subheader("AI-Powered Forensic Investigation Platform")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Case Management", "Evidence Upload", "AI Assistant", "Case Analysis"]
    )
    
    # Check API connection
    if not check_api_connection():
        st.error("❌ Cannot connect to the API server. Please ensure the backend is running on localhost:8000")
        st.info("Run: `python demo_case_platform.py` to start the backend")
        return
    
    # Route to appropriate page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Case Management":
        show_case_management()
    elif page == "Evidence Upload":
        show_evidence_upload()
    elif page == "AI Assistant":
        show_ai_assistant()
    elif page == "Case Analysis":
        show_case_analysis()

def check_api_connection() -> bool:
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def show_dashboard():
    """Dashboard with system overview"""
    st.header("📊 Investigation Dashboard")
    
    # Get cases overview
    try:
        response = requests.get(f"{API_BASE}/cases")
        if response.status_code == 200:
            cases = response.json()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cases", len(cases))
            
            with col2:
                active_cases = sum(1 for case in cases if case.get("status") == "active")
                st.metric("Active Cases", active_cases)
            
            with col3:
                total_evidence = sum(case.get("total_evidence_count", 0) for case in cases)
                st.metric("Evidence Files", total_evidence)
            
            with col4:
                processed_evidence = sum(case.get("processed_evidence_count", 0) for case in cases)
                processing_rate = (processed_evidence / total_evidence * 100) if total_evidence > 0 else 0
                st.metric("Processing Rate", f"{processing_rate:.1f}%")
            
            # Cases table
            st.subheader("Recent Cases")
            if cases:
                df = pd.DataFrame([
                    {
                        "Case Number": case.get("case_number", ""),
                        "Title": case.get("title", ""),
                        "Investigator": case.get("investigator_name", ""),
                        "Status": case.get("status", ""),
                        "Evidence": f"{case.get('processed_evidence_count', 0)}/{case.get('total_evidence_count', 0)}",
                        "Progress": f"{case.get('processing_progress', 0):.1f}%",
                        "Updated": case.get("updated_at", "").split("T")[0] if case.get("updated_at") else ""
                    }
                    for case in cases
                ])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No cases found. Create a new case to get started.")
        
        else:
            st.error("Failed to load cases")
    
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def show_case_management():
    """Case creation and management interface"""
    st.header("🗂️ Case Management")
    
    tab1, tab2 = st.tabs(["Create New Case", "Manage Existing Cases"])
    
    with tab1:
        st.subheader("Create New Investigation Case")
        
        with st.form("create_case"):
            col1, col2 = st.columns(2)
            
            with col1:
                case_number = st.text_input("Case Number*", placeholder="CASE-2024-XXX")
                title = st.text_input("Case Title*", placeholder="Brief description of the case")
                investigator_name = st.text_input("Lead Investigator*", placeholder="Detective Name")
                investigator_id = st.text_input("Investigator ID", placeholder="DET-001")
                department = st.text_input("Department", placeholder="Cybercrime Unit")
            
            with col2:
                case_type = st.selectbox("Case Type", 
                    ["", "Fraud", "Cybercrime", "Organized Crime", "Financial Crime", "Drug Crime", "Other"])
                priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
                incident_date = st.date_input("Incident Date")
                due_date = st.date_input("Due Date")
                jurisdiction = st.text_input("Jurisdiction", placeholder="City/State/Federal")
            
            description = st.text_area("Case Description", placeholder="Detailed description of the investigation...")
            tags = st.text_input("Tags (comma-separated)", placeholder="fraud, cryptocurrency, international")
            
            submit_case = st.form_submit_button("Create Case", type="primary")
            
            if submit_case:
                if case_number and title and investigator_name:
                    case_data = {
                        "case_number": case_number,
                        "title": title,
                        "investigator_name": investigator_name,
                        "description": description,
                        "investigator_id": investigator_id,
                        "department": department,
                        "case_type": case_type if case_type else None,
                        "priority": priority,
                        "jurisdiction": jurisdiction,
                        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
                    }
                    
                    try:
                        response = requests.post(f"{API_BASE}/cases", json=case_data)
                        if response.status_code == 200:
                            st.success(f"✅ Case {case_number} created successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to create case: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error creating case: {str(e)}")
                else:
                    st.error("Please fill in all required fields (*)")
    
    with tab2:
        st.subheader("Existing Cases")
        
        # Load and display cases
        try:
            response = requests.get(f"{API_BASE}/cases")
            if response.status_code == 200:
                cases = response.json()
                
                for case in cases:
                    with st.expander(f"📁 {case.get('case_number', 'Unknown')} - {case.get('title', 'No Title')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Investigator:** {case.get('investigator_name', 'Unknown')}")
                            st.write(f"**Status:** {case.get('status', 'Unknown')}")
                            st.write(f"**Priority:** {case.get('priority', 'Unknown')}")
                            st.write(f"**Created:** {case.get('created_at', '').split('T')[0] if case.get('created_at') else 'Unknown'}")
                        
                        with col2:
                            st.write(f"**Evidence Files:** {case.get('total_evidence_count', 0)}")
                            st.write(f"**Processed:** {case.get('processed_evidence_count', 0)}")
                            progress = case.get('processing_progress', 0)
                            st.progress(progress / 100 if progress <= 100 else 1.0)
                        
                        if st.button(f"View Details", key=f"view_{case.get('id')}"):
                            st.session_state.selected_case = case.get('id')
                            st.rerun()
            
        except Exception as e:
            st.error(f"Error loading cases: {str(e)}")

def show_evidence_upload():
    """Evidence upload interface"""
    st.header("📤 Evidence Upload")
    
    # Case selection
    try:
        response = requests.get(f"{API_BASE}/cases")
        if response.status_code == 200:
            cases = response.json()
            
            if not cases:
                st.warning("No cases available. Please create a case first.")
                return
            
            case_options = {f"{case['case_number']} - {case['title']}": case['id'] for case in cases}
            selected_case_display = st.selectbox("Select Case for Evidence Upload", list(case_options.keys()))
            selected_case_id = case_options[selected_case_display]
            
            st.subheader(f"Upload Evidence to: {selected_case_display}")
            
            with st.form("upload_evidence"):
                uploaded_file = st.file_uploader(
                    "Choose evidence file",
                    type=['xml', 'csv', 'json', 'txt', 'pdf', 'jpg', 'jpeg', 'png', 'mp4', 'mp3'],
                    help="Supported formats: XML, CSV, JSON, TXT, PDF, Images, Videos, Audio"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    evidence_type = st.selectbox("Evidence Type", [
                        "chat", "call_log", "contact", "document", "text_report",
                        "xml_report", "json_data", "csv_data", "image", "video", "audio"
                    ])
                    title = st.text_input("Evidence Title", placeholder="Brief description")
                    source_device = st.text_input("Source Device", placeholder="iPhone 13, Samsung Galaxy, etc.")
                
                with col2:
                    description = st.text_area("Description", placeholder="Detailed description of the evidence")
                    extraction_method = st.text_input("Extraction Method", placeholder="Cellebrite, manual export, etc.")
                
                submit_upload = st.form_submit_button("Upload Evidence", type="primary")
                
                if submit_upload and uploaded_file:
                    try:
                        # Prepare form data
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {
                            "evidence_type": evidence_type,
                            "title": title if title else uploaded_file.name,
                            "description": description,
                            "source_device": source_device,
                            "extraction_method": extraction_method
                        }
                        
                        # Upload to API
                        response = requests.post(f"{API_BASE}/cases/{selected_case_id}/evidence", files=files, data=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Evidence uploaded successfully!")
                            st.info(f"Evidence ID: {result.get('evidence_id')}")
                            st.info("🔄 Processing started in background. Check case details for progress.")
                        else:
                            st.error(f"❌ Upload failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"❌ Upload error: {str(e)}")
                
                elif submit_upload:
                    st.error("Please select a file to upload")
    
    except Exception as e:
        st.error(f"Error loading cases: {str(e)}")

def show_ai_assistant():
    """AI Assistant chatbot interface"""
    st.header("🤖 AI Investigation Assistant")
    
    # Case selection
    try:
        response = requests.get(f"{API_BASE}/cases")
        if response.status_code == 200:
            cases = response.json()
            
            if not cases:
                st.warning("No cases available. Please create a case and upload evidence first.")
                return
            
            case_options = {f"{case['case_number']} - {case['title']}": case['id'] for case in cases}
            selected_case_display = st.selectbox("Select Case for AI Analysis", list(case_options.keys()))
            selected_case_id = case_options[selected_case_display]
            
            # Get case details
            case_response = requests.get(f"{API_BASE}/cases/{selected_case_id}")
            if case_response.status_code == 200:
                case_details = case_response.json()
                
                # Case info
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Case:** {case_details['case']['case_number']}")
                    st.info(f"**Investigator:** {case_details['case']['investigator_name']}")
                
                with col2:
                    evidence_count = case_details['statistics']['processing']['total_evidence']
                    processed_count = case_details['statistics']['processing']['processed_evidence']
                    st.info(f"**Evidence:** {processed_count}/{evidence_count} processed")
                
                if processed_count == 0:
                    st.warning("⚠️ No processed evidence available. Please upload and process evidence files first.")
                    return
                
                # Chat interface
                st.subheader("💬 Chat with AI Assistant")
                
                # Get suggested questions
                suggestions_response = requests.get(f"{API_BASE}/cases/{selected_case_id}/chat/suggestions")
                if suggestions_response.status_code == 200:
                    suggestions = suggestions_response.json()
                    
                    st.write("**Suggested Questions:**")
                    cols = st.columns(2)
                    for i, suggestion in enumerate(suggestions['suggestions'][:6]):
                        col = cols[i % 2]
                        if col.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.chat_query = suggestion
                
                # Chat input
                query = st.text_input("Ask a question about the case:", 
                                    value=st.session_state.get('chat_query', ''),
                                    placeholder="What evidence do we have? Are there any suspicious activities?")
                
                if st.button("🔍 Ask AI Assistant", type="primary") and query:
                    with st.spinner("🧠 Analyzing evidence and generating response..."):
                        try:
                            chat_data = {
                                "message": query,
                                "case_id": selected_case_id
                            }
                            
                            response = requests.post(f"{API_BASE}/cases/{selected_case_id}/chat", json=chat_data)
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                # Display response
                                st.subheader("🤖 AI Response")
                                st.write(result.get('response', 'No response generated'))
                                
                                # Show confidence and metadata
                                col1, col2 = st.columns(2)
                                with col1:
                                    confidence = result.get('confidence', 0)
                                    st.metric("Confidence", f"{confidence:.1%}")
                                
                                with col2:
                                    sources_count = len(result.get('sources', []))
                                    st.metric("Sources", sources_count)
                                
                                # Show sources
                                if result.get('sources'):
                                    st.subheader("📄 Evidence Sources")
                                    for i, source in enumerate(result['sources'][:3]):
                                        with st.expander(f"Source {i+1}: {source.get('source_file', 'Unknown')}"):
                                            st.write(f"**Type:** {source.get('evidence_type', 'Unknown')}")
                                            st.write(f"**Content:** {source.get('content_preview', 'No preview')}")
                                            st.write(f"**Relevance:** {source.get('similarity_score', 0):.3f}")
                            
                            else:
                                st.error(f"❌ Query failed: {response.text}")
                        
                        except Exception as e:
                            st.error(f"❌ Error querying AI: {str(e)}")
                
                # Clear chat query from session state
                if 'chat_query' in st.session_state:
                    del st.session_state.chat_query
    
    except Exception as e:
        st.error(f"Error loading AI assistant: {str(e)}")

def show_case_analysis():
    """Case analysis and reporting interface"""
    st.header("📈 Case Analysis & Reports")
    
    # Case selection
    try:
        response = requests.get(f"{API_BASE}/cases")
        if response.status_code == 200:
            cases = response.json()
            
            if not cases:
                st.warning("No cases available.")
                return
            
            case_options = {f"{case['case_number']} - {case['title']}": case['id'] for case in cases}
            selected_case_display = st.selectbox("Select Case for Analysis", list(case_options.keys()))
            selected_case_id = case_options[selected_case_display]
            
            # Get detailed case information
            case_response = requests.get(f"{API_BASE}/cases/{selected_case_id}")
            if case_response.status_code == 200:
                case_data = case_response.json()
                case_info = case_data['case']
                statistics = case_data['statistics']
                
                # Case overview
                st.subheader("📋 Case Overview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Case Number", case_info['case_number'])
                    st.metric("Priority", case_info['priority'].title())
                    st.metric("Status", case_info['status'].title())
                
                with col2:
                    st.metric("Total Evidence", statistics['processing']['total_evidence'])
                    st.metric("Processed", statistics['processing']['processed_evidence'])
                    processing_rate = statistics['processing']['processing_rate'] * 100
                    st.metric("Processing Rate", f"{processing_rate:.1f}%")
                
                with col3:
                    st.metric("Total Chunks", statistics['processing']['total_chunks'])
                    if statistics['embeddings']['has_embeddings']:
                        st.success("✅ AI Ready")
                    else:
                        st.warning("⚠️ Processing needed")
                
                # Evidence breakdown
                st.subheader("📊 Evidence Analysis")
                
                evidence_types = statistics.get('evidence_by_type', {})
                if evidence_types:
                    # Create evidence type chart
                    evidence_df = pd.DataFrame([
                        {"Evidence Type": k.replace('_', ' ').title(), "Count": v}
                        for k, v in evidence_types.items() if v > 0
                    ])
                    
                    if not evidence_df.empty:
                        st.bar_chart(evidence_df.set_index('Evidence Type'))
                        st.dataframe(evidence_df, use_container_width=True)
                
                # Investigation timeline
                st.subheader("📅 Investigation Timeline")
                timeline_data = [
                    {"Event": "Case Created", "Date": case_info['created_at'].split('T')[0]},
                    {"Event": "Last Updated", "Date": case_info['updated_at'].split('T')[0]},
                ]
                
                if case_info.get('incident_date'):
                    timeline_data.insert(0, {"Event": "Incident Date", "Date": case_info['incident_date']})
                
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df, use_container_width=True)
                
                # Case details
                st.subheader("📝 Case Details")
                
                with st.expander("Full Case Information", expanded=False):
                    st.json(case_info)
    
    except Exception as e:
        st.error(f"Error loading case analysis: {str(e)}")

if __name__ == "__main__":
    main()