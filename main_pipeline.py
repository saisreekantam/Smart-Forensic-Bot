"""
Main Processing Pipeline for Project Sentinel
Demonstrates the complete data preprocessing and chunking pipeline
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

from data_ingestion import UFDRDataIngestion
from data_ingestion.preprocessor import DataPreprocessor  
from data_ingestion.chunking import ChunkManager
from config.settings import settings
from src.utils.data_generator import UFDRSampleDataGenerator

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProjectSentinelPipeline:
    """Main processing pipeline for Project Sentinel"""
    
    def __init__(self):
        self.data_ingestion = UFDRDataIngestion()
        self.preprocessor = DataPreprocessor()
        self.chunk_manager = ChunkManager()
        
    def process_single_file(self, file_path: str) -> dict:
        """Process a single UFDR file through the complete pipeline"""
        
        logger.info(f"Starting processing of file: {file_path}")
        
        try:
            # Step 1: Parse the UFDR file
            logger.info("Step 1: Parsing UFDR file...")
            ufdr_document = self.data_ingestion.process_file(file_path)
            logger.info(f"Successfully parsed {ufdr_document.file_type} file")
            
            # Step 2: Preprocess the content
            logger.info("Step 2: Preprocessing content...")
            preprocessed_content = self.preprocessor.preprocess_document(ufdr_document.content)
            logger.info(f"Preprocessing complete. Found {preprocessed_content['metadata']['total_entities']} entities")
            
            # Step 3: Create text chunks
            logger.info("Step 3: Creating text chunks...")
            chunks = self.chunk_manager.process_document_for_chunking(
                preprocessed_content, 
                ufdr_document.file_path
            )
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Step 4: Optimize chunks for embedding
            logger.info("Step 4: Optimizing chunks for embeddings...")
            optimized_chunks = self.chunk_manager.optimize_chunks_for_embedding(chunks)
            logger.info(f"Optimized to {len(optimized_chunks)} chunks")
            
            # Get statistics
            chunk_stats = self.chunk_manager.get_chunk_statistics(optimized_chunks)
            
            return {
                "status": "success",
                "file_path": file_path,
                "ufdr_document": ufdr_document,
                "preprocessed_content": preprocessed_content,
                "chunks": optimized_chunks,
                "statistics": {
                    "chunk_stats": chunk_stats,
                    "total_entities": preprocessed_content['metadata']['total_entities'],
                    "processing_time": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "status": "error",
                "file_path": file_path,
                "error": str(e)
            }
    
    def process_directory(self, directory_path: str) -> dict:
        """Process all UFDR files in a directory"""
        
        logger.info(f"Starting batch processing of directory: {directory_path}")
        
        results = {
            "status": "success",
            "directory": directory_path,
            "processed_files": [],
            "failed_files": [],
            "summary": {}
        }
        
        try:
            # Get all supported files
            directory = Path(directory_path)
            supported_extensions = settings.supported_formats
            
            files_to_process = []
            for ext in supported_extensions:
                files_to_process.extend(list(directory.glob(f"*.{ext}")))
            
            logger.info(f"Found {len(files_to_process)} files to process")
            
            total_chunks = 0
            total_entities = 0
            
            # Process each file
            for file_path in files_to_process:
                logger.info(f"Processing file: {file_path}")
                
                result = self.process_single_file(str(file_path))
                
                if result["status"] == "success":
                    results["processed_files"].append(result)
                    total_chunks += len(result["chunks"])
                    total_entities += result["statistics"]["total_entities"]
                else:
                    results["failed_files"].append(result)
            
            # Summary statistics
            results["summary"] = {
                "total_files_attempted": len(files_to_process),
                "successful_files": len(results["processed_files"]),
                "failed_files": len(results["failed_files"]),
                "total_chunks_created": total_chunks,
                "total_entities_extracted": total_entities,
                "average_chunks_per_file": total_chunks / len(results["processed_files"]) if results["processed_files"] else 0
            }
            
            logger.info(f"Batch processing complete. {results['summary']}")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def save_results(self, results: dict, output_file: str = None) -> str:
        """Save processing results to file"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = settings.processed_data_dir / f"processing_results_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert chunks to dictionaries for JSON serialization
        if "chunks" in results:
            results["chunks"] = [chunk.to_dict() for chunk in results["chunks"]]
        elif "processed_files" in results:
            for file_result in results["processed_files"]:
                if "chunks" in file_result:
                    file_result["chunks"] = [chunk.to_dict() for chunk in file_result["chunks"]]
        
        # Convert UFDR documents to dictionaries
        def convert_ufdr_document(doc):
            return {
                "file_path": doc.file_path,
                "file_type": doc.file_type,
                "case_id": doc.case_id,
                "device_info": doc.device_info,
                "content": doc.content,
                "metadata": doc.metadata,
                "extracted_at": doc.extracted_at.isoformat()
            }
        
        if "ufdr_document" in results:
            results["ufdr_document"] = convert_ufdr_document(results["ufdr_document"])
        elif "processed_files" in results:
            for file_result in results["processed_files"]:
                if "ufdr_document" in file_result:
                    file_result["ufdr_document"] = convert_ufdr_document(file_result["ufdr_document"])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        return str(output_path)
    
    def demonstrate_pipeline(self) -> None:
        """Demonstrate the complete pipeline with sample data"""
        
        logger.info("=== Project Sentinel Pipeline Demonstration ===")
        
        # Step 1: Generate sample data if needed
        sample_dir = settings.sample_data_dir
        if not any(Path(sample_dir).glob("*")):
            logger.info("Generating sample data...")
            generator = UFDRSampleDataGenerator()
            generator.save_sample_data()
        
        # Step 2: Process all sample files
        logger.info("Processing all sample files...")
        results = self.process_directory(str(sample_dir))
        
        # Step 3: Save results
        output_file = self.save_results(results)
        
        # Step 4: Display summary
        self._display_summary(results)
        
        logger.info(f"Demonstration complete. Results saved to: {output_file}")
    
    def _display_summary(self, results: dict) -> None:
        """Display processing summary"""
        
        print("\n" + "="*60)
        print("PROJECT SENTINEL - PROCESSING SUMMARY")
        print("="*60)
        
        if results["status"] == "success":
            summary = results["summary"]
            print(f"📁 Directory processed: {results['directory']}")
            print(f"📄 Files attempted: {summary['total_files_attempted']}")
            print(f"✅ Successful: {summary['successful_files']}")
            print(f"❌ Failed: {summary['failed_files']}")
            print(f"🧩 Total chunks created: {summary['total_chunks_created']}")
            print(f"🏷️  Total entities extracted: {summary['total_entities_extracted']}")
            print(f"📊 Average chunks per file: {summary['average_chunks_per_file']:.1f}")
            
            print("\n📋 FILES PROCESSED:")
            for i, file_result in enumerate(results["processed_files"], 1):
                chunks_count = len(file_result["chunks"])
                entities_count = file_result["statistics"]["total_entities"]
                file_name = Path(file_result["file_path"]).name
                print(f"  {i}. {file_name} - {chunks_count} chunks, {entities_count} entities")
            
            if results["failed_files"]:
                print("\n❌ FAILED FILES:")
                for i, failed in enumerate(results["failed_files"], 1):
                    file_name = Path(failed["file_path"]).name
                    print(f"  {i}. {file_name} - Error: {failed['error']}")
        
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        
        print("="*60)
        print("🎯 Project Sentinel: Transforming forensic investigation with AI!")
        print("="*60)

def main():
    """Main entry point"""
    
    # Create pipeline
    pipeline = ProjectSentinelPipeline()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        if Path(path).is_file():
            # Process single file
            logger.info(f"Processing single file: {path}")
            result = pipeline.process_single_file(path)
            output_file = pipeline.save_results(result)
            
            if result["status"] == "success":
                print(f"✅ Successfully processed {path}")
                print(f"📊 Created {len(result['chunks'])} chunks")
                print(f"🏷️  Extracted {result['statistics']['total_entities']} entities")
                print(f"💾 Results saved to: {output_file}")
            else:
                print(f"❌ Failed to process {path}: {result['error']}")
                
        elif Path(path).is_dir():
            # Process directory
            logger.info(f"Processing directory: {path}")
            results = pipeline.process_directory(path)
            output_file = pipeline.save_results(results)
            pipeline._display_summary(results)
            
        else:
            print(f"❌ Path not found: {path}")
    else:
        # Run demonstration
        pipeline.demonstrate_pipeline()

if __name__ == "__main__":
    main()