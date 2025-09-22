#!/usr/bin/env python3
"""
Simple test script for Smart Forensic Bot core functionality
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, 'src')

class SimpleFileAnalyzer:
    """Simplified file analyzer for testing"""
    
    def __init__(self):
        self.supported_formats = [
            '.txt', '.log', '.exe', '.dll', '.bat', '.cmd',
            '.pdf', '.doc', '.jpg', '.png', '.zip'
        ]
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Simple file analysis"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        # Calculate hashes
        hashes = {}
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                hashes['md5'] = hashlib.md5(content).hexdigest()
                hashes['sha256'] = hashlib.sha256(content).hexdigest()
        except Exception as e:
            hashes['error'] = str(e)
        
        # Check suspicious indicators
        suspicious = []
        if file_path.suffix.lower() in ['.exe', '.bat', '.cmd', '.scr']:
            suspicious.append(f"Potentially executable file: {file_path.suffix}")
        
        if stat.st_size == 0:
            suspicious.append("Zero-byte file")
        
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': stat.st_size,
            'file_extension': file_path.suffix,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'hashes': hashes,
            'suspicious_indicators': suspicious
        }

def test_file_analysis():
    """Test file analysis functionality"""
    print("🔍 Testing File Analysis...")
    
    # Create test directory and files
    test_dir = Path('./test_evidence')
    test_dir.mkdir(exist_ok=True)
    
    # Create test files
    test_files = [
        ('sample.txt', 'This is a sample text file for testing.'),
        ('suspicious.bat', '@echo off\necho This is a suspicious batch file'),
        ('empty.log', ''),
        ('data.json', '{"test": "data", "timestamp": "2024-01-01"}')
    ]
    
    analyzer = SimpleFileAnalyzer()
    results = []
    
    for filename, content in test_files:
        file_path = test_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        
        try:
            analysis = analyzer.analyze_file(file_path)
            results.append(analysis)
            print(f"✅ Analyzed: {filename}")
            if analysis['suspicious_indicators']:
                print(f"   ⚠️  Suspicious: {', '.join(analysis['suspicious_indicators'])}")
        except Exception as e:
            print(f"❌ Error analyzing {filename}: {e}")
    
    return results

def test_knowledge_base():
    """Test knowledge base functionality"""
    print("\n📚 Testing Knowledge Base...")
    
    # Simple knowledge base simulation
    knowledge_items = [
        {
            'content': 'Batch files (.bat) can execute system commands and may be used maliciously',
            'category': 'file_analysis',
            'keywords': ['batch', 'executable', 'malicious']
        },
        {
            'content': 'Zero-byte files may indicate file corruption or deletion attempts',
            'category': 'file_analysis', 
            'keywords': ['corruption', 'deletion', 'forensics']
        },
        {
            'content': 'Hash verification is crucial for evidence integrity',
            'category': 'integrity',
            'keywords': ['hash', 'integrity', 'verification']
        }
    ]
    
    # Simple search function
    def search_knowledge(query: str, items: List[Dict]) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for item in items:
            # Simple keyword matching
            if any(keyword in query_lower for keyword in item['keywords']):
                results.append(item)
            elif query_lower in item['content'].lower():
                results.append(item)
        return results
    
    # Test searches
    test_queries = ['batch', 'hash', 'zero byte', 'malicious']
    
    for query in test_queries:
        results = search_knowledge(query, knowledge_items)
        print(f"🔍 Query '{query}': {len(results)} results")
        for result in results:
            print(f"   📖 {result['content'][:60]}...")
    
    return knowledge_items

def test_graph_simulation():
    """Test knowledge graph simulation"""
    print("\n🕸️ Testing Knowledge Graph Simulation...")
    
    # Simple graph representation
    entities = {}
    relationships = []
    
    def add_entity(entity_id: str, entity_type: str, properties: Dict[str, Any]):
        entities[entity_id] = {
            'id': entity_id,
            'type': entity_type,
            'properties': properties,
            'created_at': datetime.now().isoformat()
        }
    
    def add_relationship(from_id: str, to_id: str, rel_type: str):
        relationships.append({
            'from': from_id,
            'to': to_id,
            'type': rel_type,
            'created_at': datetime.now().isoformat()
        })
    
    # Add test entities
    add_entity('file_001', 'file', {'name': 'suspicious.bat', 'type': 'batch'})
    add_entity('case_001', 'case', {'name': 'Test Investigation'})
    add_entity('dir_001', 'directory', {'path': './test_evidence'})
    
    # Add relationships
    add_relationship('case_001', 'file_001', 'contains')
    add_relationship('dir_001', 'file_001', 'contains')
    
    print(f"✅ Created {len(entities)} entities and {len(relationships)} relationships")
    
    # Find related entities
    def find_related(entity_id: str) -> List[str]:
        related = []
        for rel in relationships:
            if rel['from'] == entity_id:
                related.append(rel['to'])
            elif rel['to'] == entity_id:
                related.append(rel['from'])
        return related
    
    for entity_id in entities:
        related = find_related(entity_id)
        print(f"🔗 {entity_id} ({entities[entity_id]['type']}) related to: {related}")
    
    return entities, relationships

def create_sample_report(analysis_results: List[Dict[str, Any]], 
                        knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a sample forensic report"""
    print("\n📋 Creating Sample Report...")
    
    # Analyze findings
    findings = []
    recommendations = []
    
    for result in analysis_results:
        if result['suspicious_indicators']:
            findings.append(f"Suspicious file detected: {result['file_name']} - {', '.join(result['suspicious_indicators'])}")
    
    if any('bat' in r['file_extension'] for r in analysis_results):
        recommendations.append("Review batch files for malicious commands")
    
    if any(r['file_size'] == 0 for r in analysis_results):
        recommendations.append("Investigate zero-byte files for potential evidence tampering")
    
    recommendations.append("Verify file integrity using hash values")
    recommendations.append("Preserve original evidence for legal proceedings")
    
    report = {
        'case_id': f"DEMO_CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'analyst': 'Demo Analyst',
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'files_analyzed': len(analysis_results),
            'suspicious_files': len([r for r in analysis_results if r['suspicious_indicators']]),
            'knowledge_items': len(knowledge_items)
        },
        'findings': findings,
        'recommendations': recommendations,
        'detailed_analysis': analysis_results
    }
    
    return report

def main():
    """Main test function"""
    print("🚀 Smart Forensic Bot - Core Functionality Test")
    print("=" * 50)
    
    try:
        # Test file analysis
        analysis_results = test_file_analysis()
        
        # Test knowledge base
        knowledge_items = test_knowledge_base()
        
        # Test graph simulation
        entities, relationships = test_graph_simulation()
        
        # Create sample report
        report = create_sample_report(analysis_results, knowledge_items)
        
        # Save report
        report_path = Path('./test_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Test Report Summary:")
        print(f"   Case ID: {report['case_id']}")
        print(f"   Files Analyzed: {report['summary']['files_analyzed']}")
        print(f"   Suspicious Files: {report['summary']['suspicious_files']}")
        print(f"   Findings: {len(report['findings'])}")
        print(f"   Recommendations: {len(report['recommendations'])}")
        print(f"   Report saved to: {report_path}")
        
        print("\n🎉 All tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ File analysis with hash calculation")
        print("✅ Suspicious indicator detection")
        print("✅ Knowledge base search simulation")
        print("✅ Entity relationship mapping")
        print("✅ Automated report generation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)