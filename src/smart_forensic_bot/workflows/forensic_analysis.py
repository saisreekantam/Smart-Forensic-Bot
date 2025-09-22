"""
LangGraph workflow for forensic analysis
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from ..core.config import config
from ..forensics.file_analyzer import FileAnalyzer
from ..rag.vector_store import ForensicVectorStore
from ..knowledge_graph.graph import ForensicKnowledgeGraph

class ForensicAnalysisState:
    """State class for forensic analysis workflow"""
    
    def __init__(self):
        self.evidence_files: List[str] = []
        self.analysis_results: Dict[str, Any] = {}
        self.findings: List[str] = []
        self.recommendations: List[str] = []
        self.knowledge_context: List[str] = []
        self.current_step: str = "initial"
        self.metadata: Dict[str, Any] = {
            'case_id': '',
            'analyst': '',
            'timestamp': datetime.now().isoformat()
        }

class ForensicWorkflow:
    """LangGraph workflow for automated forensic analysis"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.openai_api_key
        )
        
        self.file_analyzer = FileAnalyzer()
        self.vector_store = ForensicVectorStore()
        self.knowledge_graph = ForensicKnowledgeGraph()
        
        # Initialize LangGraph
        self.workflow = self._create_workflow()
        
        # Setup checkpointing
        self.checkpointer = SqliteSaver.from_conn_string(
            str(config.langgraph_checkpoint_path / "checkpoints.sqlite")
        )
        
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ForensicAnalysisState)
        
        # Add nodes
        workflow.add_node("analyze_files", self._analyze_files_node)
        workflow.add_node("extract_knowledge", self._extract_knowledge_node)
        workflow.add_node("correlate_evidence", self._correlate_evidence_node)
        workflow.add_node("generate_findings", self._generate_findings_node)
        workflow.add_node("create_recommendations", self._create_recommendations_node)
        workflow.add_node("update_knowledge_graph", self._update_knowledge_graph_node)
        
        # Define the workflow
        workflow.set_entry_point("analyze_files")
        workflow.add_edge("analyze_files", "extract_knowledge")
        workflow.add_edge("extract_knowledge", "correlate_evidence")
        workflow.add_edge("correlate_evidence", "generate_findings")
        workflow.add_edge("generate_findings", "create_recommendations")
        workflow.add_edge("create_recommendations", "update_knowledge_graph")
        workflow.add_edge("update_knowledge_graph", END)
        
        return workflow
    
    def _analyze_files_node(self, state: ForensicAnalysisState) -> ForensicAnalysisState:
        """Analyze evidence files"""
        state.current_step = "analyzing_files"
        
        for file_path in state.evidence_files:
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    analysis = self.file_analyzer.analyze_file(file_path_obj)
                    state.analysis_results[file_path] = analysis
                    
                    # Add to knowledge graph
                    self.knowledge_graph.add_forensic_evidence({
                        'id': f"file_{analysis['hashes']['sha256'][:8]}",
                        'name': analysis['file_name'],
                        'file_path': file_path,
                        'hash_md5': analysis['hashes']['md5'],
                        'hash_sha256': analysis['hashes']['sha256'],
                        'file_type': analysis['file_type']['mime_type'],
                        'size': analysis['file_size'],
                        'created_time': analysis['created_time'],
                        'modified_time': analysis['modified_time'],
                        'case_id': state.metadata.get('case_id', 'unknown')
                    })
                    
            except Exception as e:
                state.analysis_results[file_path] = {'error': str(e)}
        
        return state
    
    def _extract_knowledge_node(self, state: ForensicAnalysisState) -> ForensicAnalysisState:
        """Extract relevant knowledge from vector store"""
        state.current_step = "extracting_knowledge"
        
        # Create search queries based on analysis results
        search_queries = []
        
        for file_path, analysis in state.analysis_results.items():
            if 'error' not in analysis:
                # Search for similar file types
                file_type = analysis.get('file_type', {}).get('mime_type', '')
                if file_type:
                    search_queries.append(f"forensic analysis {file_type}")
                
                # Search for suspicious indicators
                indicators = analysis.get('suspicious_indicators', [])
                for indicator in indicators:
                    search_queries.append(f"forensic investigation {indicator}")
        
        # Retrieve relevant knowledge
        knowledge_context = []
        for query in search_queries[:5]:  # Limit to top 5 queries
            results = self.vector_store.search(query, k=3)
            for doc, score, metadata in results:
                if score < 1.0:  # Only include relevant results
                    knowledge_context.append(doc)
        
        state.knowledge_context = knowledge_context
        return state
    
    def _correlate_evidence_node(self, state: ForensicAnalysisState) -> ForensicAnalysisState:
        """Correlate evidence using knowledge graph"""
        state.current_step = "correlating_evidence"
        
        # Find relationships between analyzed files
        correlations = []
        
        for file_path, analysis in state.analysis_results.items():
            if 'error' not in analysis:
                file_id = f"file_{analysis['hashes']['sha256'][:8]}"
                related_entities = self.knowledge_graph.find_related_entities(file_id)
                
                if related_entities:
                    correlations.append({
                        'file_path': file_path,
                        'related_entities': related_entities
                    })
        
        state.analysis_results['correlations'] = correlations
        return state
    
    def _generate_findings_node(self, state: ForensicAnalysisState) -> ForensicAnalysisState:
        """Generate forensic findings using LLM"""
        state.current_step = "generating_findings"
        
        # Prepare context for LLM
        analysis_summary = self._prepare_analysis_summary(state)
        knowledge_context = "\n".join(state.knowledge_context[:5])
        
        prompt = f"""
        You are a digital forensics expert analyzing evidence. Based on the following analysis results and knowledge base, provide key findings:

        ANALYSIS RESULTS:
        {analysis_summary}

        RELEVANT KNOWLEDGE:
        {knowledge_context}

        Please provide:
        1. Key findings from the analysis
        2. Potential security threats or incidents identified
        3. Evidence of malicious activity
        4. Timeline of events (if determinable)

        Format your response as a structured list of findings.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            findings_text = response.content
            
            # Parse findings (simple line-based parsing)
            findings = [line.strip() for line in findings_text.split('\n') 
                       if line.strip() and not line.strip().startswith('#')]
            
            state.findings = findings
            
        except Exception as e:
            state.findings = [f"Error generating findings: {str(e)}"]
        
        return state
    
    def _create_recommendations_node(self, state: ForensicAnalysisState) -> ForensicAnalysisState:
        """Create recommendations based on findings"""
        state.current_step = "creating_recommendations"
        
        findings_text = "\n".join(state.findings)
        
        prompt = f"""
        Based on the following forensic findings, provide actionable recommendations for investigation and remediation:

        FINDINGS:
        {findings_text}

        Please provide:
        1. Immediate actions to take
        2. Further investigation steps
        3. Security measures to implement
        4. Evidence preservation recommendations

        Format your response as a structured list of recommendations.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            recommendations_text = response.content
            
            # Parse recommendations
            recommendations = [line.strip() for line in recommendations_text.split('\n') 
                             if line.strip() and not line.strip().startswith('#')]
            
            state.recommendations = recommendations
            
        except Exception as e:
            state.recommendations = [f"Error generating recommendations: {str(e)}"]
        
        return state
    
    def _update_knowledge_graph_node(self, state: ForensicAnalysisState) -> ForensicAnalysisState:
        """Update knowledge graph with findings"""
        state.current_step = "updating_knowledge_graph"
        
        # Add case findings to vector store for future reference
        case_data = {
            'case_id': state.metadata.get('case_id', 'unknown'),
            'summary': f"Forensic analysis of {len(state.evidence_files)} files",
            'findings': state.findings,
            'evidence': [
                {
                    'id': f"file_{analysis.get('hashes', {}).get('sha256', 'unknown')[:8]}",
                    'description': f"File: {analysis.get('file_name', 'unknown')} - {analysis.get('file_type', {}).get('description', 'unknown')}"
                }
                for analysis in state.analysis_results.values()
                if 'error' not in analysis and 'hashes' in analysis
            ]
        }
        
        self.vector_store.add_forensic_case(case_data)
        
        state.current_step = "completed"
        return state
    
    def _prepare_analysis_summary(self, state: ForensicAnalysisState) -> str:
        """Prepare a summary of analysis results for LLM consumption"""
        summary_parts = []
        
        for file_path, analysis in state.analysis_results.items():
            if 'error' in analysis:
                summary_parts.append(f"ERROR analyzing {file_path}: {analysis['error']}")
            else:
                summary_parts.append(f"""
File: {analysis.get('file_name', 'unknown')}
Path: {file_path}
Type: {analysis.get('file_type', {}).get('description', 'unknown')}
Size: {analysis.get('file_size', 0)} bytes
MD5: {analysis.get('hashes', {}).get('md5', 'unknown')}
SHA256: {analysis.get('hashes', {}).get('sha256', 'unknown')}
Suspicious Indicators: {', '.join(analysis.get('suspicious_indicators', []))}
""")
        
        # Add correlation information
        if 'correlations' in state.analysis_results:
            summary_parts.append("\nCORRELATIONS:")
            for correlation in state.analysis_results['correlations']:
                related_count = len(correlation['related_entities'])
                summary_parts.append(f"{correlation['file_path']}: {related_count} related entities found")
        
        return "\n".join(summary_parts)
    
    def run_analysis(self, evidence_files: List[str], case_id: str = None, 
                    analyst: str = None) -> Dict[str, Any]:
        """
        Run the complete forensic analysis workflow
        
        Args:
            evidence_files: List of file paths to analyze
            case_id: Optional case identifier
            analyst: Optional analyst name
            
        Returns:
            Complete analysis results
        """
        # Initialize state
        initial_state = ForensicAnalysisState()
        initial_state.evidence_files = evidence_files
        initial_state.metadata['case_id'] = case_id or f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        initial_state.metadata['analyst'] = analyst or "automated"
        
        # Run the workflow
        try:
            final_state = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state.metadata['case_id']}}
            )
            
            return {
                'case_id': final_state.metadata['case_id'],
                'analyst': final_state.metadata['analyst'],
                'timestamp': final_state.metadata['timestamp'],
                'evidence_files': final_state.evidence_files,
                'analysis_results': final_state.analysis_results,
                'findings': final_state.findings,
                'recommendations': final_state.recommendations,
                'knowledge_context': final_state.knowledge_context,
                'status': 'completed' if final_state.current_step == 'completed' else 'error'
            }
            
        except Exception as e:
            return {
                'case_id': initial_state.metadata['case_id'],
                'error': str(e),
                'status': 'error'
            }