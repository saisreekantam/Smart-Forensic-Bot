"""
Command Line Interface for Smart Forensic Bot
"""

import click
import json
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from ..workflows.forensic_analysis import ForensicWorkflow
from ..forensics.file_analyzer import FileAnalyzer
from ..rag.vector_store import ForensicVectorStore
from ..knowledge_graph.graph import ForensicKnowledgeGraph
from ..core.config import config

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Smart Forensic Bot - AI-powered digital forensics analysis system"""
    pass

@cli.command()
@click.argument('evidence_path', type=click.Path(exists=True))
@click.option('--case-id', help='Case identifier')
@click.option('--analyst', help='Analyst name')
@click.option('--output', type=click.Path(), help='Output file for results')
@click.option('--recursive', is_flag=True, help='Analyze directories recursively')
def analyze(evidence_path: str, case_id: str, analyst: str, output: str, recursive: bool):
    """Analyze evidence files or directories"""
    
    console.print(Panel.fit("🔍 Smart Forensic Bot Analysis", style="bold blue"))
    
    evidence_path_obj = Path(evidence_path)
    evidence_files = []
    
    # Collect evidence files
    if evidence_path_obj.is_file():
        evidence_files = [str(evidence_path_obj)]
    elif evidence_path_obj.is_dir():
        if recursive:
            evidence_files = [str(f) for f in evidence_path_obj.rglob('*') if f.is_file()]
        else:
            evidence_files = [str(f) for f in evidence_path_obj.iterdir() if f.is_file()]
    
    if not evidence_files:
        console.print("❌ No evidence files found", style="red")
        return
    
    console.print(f"📁 Found {len(evidence_files)} evidence files")
    
    # Initialize workflow
    with console.status("[bold green]Initializing forensic analysis workflow..."):
        workflow = ForensicWorkflow()
    
    # Run analysis
    console.print("🚀 Starting forensic analysis...")
    
    try:
        results = workflow.run_analysis(
            evidence_files=evidence_files,
            case_id=case_id,
            analyst=analyst
        )
        
        if results['status'] == 'completed':
            console.print("✅ Analysis completed successfully!", style="green")
            _display_results(results)
            
            # Save results if output specified
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"💾 Results saved to {output_path}")
        else:
            console.print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}", style="red")
            
    except Exception as e:
        console.print(f"❌ Analysis failed: {str(e)}", style="red")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output file for analysis')
def scan(file_path: str, output: str):
    """Quick scan of a single file"""
    
    console.print(Panel.fit("🔎 Quick File Scan", style="bold cyan"))
    
    file_path_obj = Path(file_path)
    
    with console.status("[bold green]Analyzing file..."):
        analyzer = FileAnalyzer()
        analysis = analyzer.analyze_file(file_path_obj)
    
    _display_file_analysis(analysis)
    
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        console.print(f"💾 Analysis saved to {output_path}")

@cli.command()
@click.argument('query')
@click.option('--limit', default=5, help='Number of results to return')
def search(query: str, limit: int):
    """Search the forensic knowledge base"""
    
    console.print(Panel.fit("🔍 Knowledge Base Search", style="bold magenta"))
    
    with console.status("[bold green]Searching knowledge base..."):
        vector_store = ForensicVectorStore()
        results = vector_store.search(query, k=limit)
    
    if results:
        table = Table(title="Search Results")
        table.add_column("Score", style="cyan")
        table.add_column("Content", style="white")
        table.add_column("Type", style="green")
        
        for content, score, metadata in results:
            table.add_row(
                f"{score:.3f}",
                content[:100] + "..." if len(content) > 100 else content,
                metadata.get('type', 'unknown')
            )
        
        console.print(table)
    else:
        console.print("No results found", style="yellow")

@cli.command()
@click.option('--entity-type', help='Filter by entity type')
def graph_stats(entity_type: str):
    """Show knowledge graph statistics"""
    
    console.print(Panel.fit("📊 Knowledge Graph Statistics", style="bold green"))
    
    graph = ForensicKnowledgeGraph()
    stats = graph.get_stats()
    
    # Overall stats
    table = Table(title="Graph Overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Nodes", str(stats['total_nodes']))
    table.add_row("Total Edges", str(stats['total_edges']))
    table.add_row("Connected", "Yes" if stats['is_connected'] else "No")
    
    console.print(table)
    
    # Node types
    if stats['node_types']:
        node_table = Table(title="Node Types")
        node_table.add_column("Type", style="cyan")
        node_table.add_column("Count", style="white")
        
        for node_type, count in stats['node_types'].items():
            node_table.add_row(node_type, str(count))
        
        console.print(node_table)
    
    # Edge types
    if stats['edge_types']:
        edge_table = Table(title="Edge Types")
        edge_table.add_column("Type", style="cyan")
        edge_table.add_column("Count", style="white")
        
        for edge_type, count in stats['edge_types'].items():
            edge_table.add_row(edge_type, str(count))
        
        console.print(edge_table)

@cli.command()
@click.argument('knowledge_file', type=click.Path(exists=True))
def import_knowledge(knowledge_file: str):
    """Import knowledge base from JSON file"""
    
    console.print(Panel.fit("📚 Importing Knowledge Base", style="bold yellow"))
    
    knowledge_path = Path(knowledge_file)
    
    try:
        with open(knowledge_path, 'r') as f:
            knowledge_data = json.load(f)
        
        vector_store = ForensicVectorStore()
        
        if isinstance(knowledge_data, list):
            with console.status("[bold green]Importing knowledge..."):
                vector_store.add_forensic_knowledge(knowledge_data)
            console.print(f"✅ Imported {len(knowledge_data)} knowledge items")
        else:
            console.print("❌ Invalid knowledge file format. Expected list of knowledge items.", style="red")
            
    except Exception as e:
        console.print(f"❌ Import failed: {str(e)}", style="red")

@cli.command()
def init():
    """Initialize Smart Forensic Bot configuration"""
    
    console.print(Panel.fit("⚙️ Smart Forensic Bot Initialization", style="bold blue"))
    
    # Create directories
    config.create_directories()
    
    console.print("✅ Created directory structure")
    
    # Check for .env file
    env_path = Path('.env')
    if not env_path.exists():
        console.print("📝 Creating .env file from template...")
        example_env = Path('.env.example')
        if example_env.exists():
            import shutil
            shutil.copy(example_env, env_path)
            console.print("✅ Created .env file. Please configure your API keys and settings.")
        else:
            console.print("❌ .env.example file not found", style="red")
    else:
        console.print("✅ .env file already exists")
    
    console.print("\n🎉 Smart Forensic Bot initialized successfully!")
    console.print("Please edit the .env file to configure your API keys and settings.")

def _display_results(results: dict):
    """Display analysis results in a formatted way"""
    
    # Case information
    info_table = Table(title="Case Information")
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Case ID", results.get('case_id', 'N/A'))
    info_table.add_row("Analyst", results.get('analyst', 'N/A'))
    info_table.add_row("Timestamp", results.get('timestamp', 'N/A'))
    info_table.add_row("Files Analyzed", str(len(results.get('evidence_files', []))))
    
    console.print(info_table)
    
    # Findings
    if results.get('findings'):
        console.print("\n🔍 Key Findings:", style="bold red")
        for i, finding in enumerate(results['findings'], 1):
            console.print(f"{i}. {finding}")
    
    # Recommendations
    if results.get('recommendations'):
        console.print("\n💡 Recommendations:", style="bold green")
        for i, recommendation in enumerate(results['recommendations'], 1):
            console.print(f"{i}. {recommendation}")

def _display_file_analysis(analysis: dict):
    """Display file analysis results"""
    
    # Basic file info
    info_table = Table(title="File Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("File Name", analysis.get('file_name', 'N/A'))
    info_table.add_row("File Size", f"{analysis.get('file_size', 0):,} bytes")
    info_table.add_row("File Type", analysis.get('file_type', {}).get('description', 'N/A'))
    info_table.add_row("MIME Type", analysis.get('file_type', {}).get('mime_type', 'N/A'))
    
    console.print(info_table)
    
    # Hashes
    hashes = analysis.get('hashes', {})
    if hashes:
        hash_table = Table(title="File Hashes")
        hash_table.add_column("Algorithm", style="cyan")
        hash_table.add_column("Hash", style="white")
        
        for algo, hash_value in hashes.items():
            hash_table.add_row(algo.upper(), hash_value)
        
        console.print(hash_table)
    
    # Suspicious indicators
    indicators = analysis.get('suspicious_indicators', [])
    if indicators:
        console.print("\n⚠️ Suspicious Indicators:", style="bold red")
        for indicator in indicators:
            console.print(f"• {indicator}")

if __name__ == '__main__':
    cli()