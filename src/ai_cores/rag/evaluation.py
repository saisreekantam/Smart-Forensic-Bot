"""
RAG Evaluation and Testing Framework for Forensic Analysis

This module provides comprehensive evaluation metrics and testing capabilities
for the forensic RAG system, including retrieval quality, generation accuracy,
and end-to-end system performance assessment.
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import statistics
from pathlib import Path

# Evaluation metrics
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Local imports
from .rag_system import ForensicRAGSystem, RAGQuery, RAGResponse
from .retrieval import RankedResult
from .generation import GeneratedResponse

logger = logging.getLogger(__name__)

@dataclass
class RetrievalGroundTruth:
    """Ground truth for retrieval evaluation"""
    query: str
    relevant_document_ids: Set[str]
    highly_relevant_ids: Set[str]  # Subset of relevant with high relevance
    query_type: str = "general"
    expected_data_types: Optional[List[str]] = None

@dataclass
class GenerationGroundTruth:
    """Ground truth for generation evaluation"""
    query: str
    reference_answer: str
    key_facts: List[str]
    expected_entities: Optional[List[str]] = None
    answer_type: str = "factual"  # 'factual', 'analytical', 'summary'

@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    ndcg_at_k: Dict[int, float]
    map_score: float  # Mean Average Precision
    coverage: float
    diversity: float

@dataclass
class GenerationMetrics:
    """Generation evaluation metrics"""
    bleu_scores: Dict[int, float]  # BLEU-1, BLEU-2, etc.
    rouge_scores: Dict[str, float]  # ROUGE-L, ROUGE-1, etc.
    semantic_similarity: float
    fact_accuracy: float
    entity_coverage: float
    hallucination_score: float
    confidence_calibration: float

@dataclass
class EndToEndMetrics:
    """End-to-end system evaluation metrics"""
    response_time: float
    throughput: float  # queries per second
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    user_satisfaction_score: Optional[float] = None
    error_rate: float = 0.0

class RetrievalEvaluator:
    """Evaluator for retrieval system performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def evaluate_retrieval(
        self,
        results: List[RankedResult],
        ground_truth: RetrievalGroundTruth,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance against ground truth
        
        Args:
            results: Retrieved and ranked results
            ground_truth: Ground truth relevance judgments
            k_values: K values for evaluation metrics
        
        Returns:
            Comprehensive retrieval metrics
        """
        
        # Extract retrieved document IDs
        retrieved_ids = [result.result.id for result in results]
        relevant_ids = ground_truth.relevant_document_ids
        highly_relevant_ids = ground_truth.highly_relevant_ids
        
        # Calculate metrics at different k values
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        ndcg_at_k = {}
        
        for k in k_values:
            if k <= len(retrieved_ids):
                retrieved_k = set(retrieved_ids[:k])
                
                # Precision@K
                precision_at_k[k] = len(retrieved_k.intersection(relevant_ids)) / k
                
                # Recall@K
                recall_at_k[k] = len(retrieved_k.intersection(relevant_ids)) / max(len(relevant_ids), 1)
                
                # F1@K
                p_k = precision_at_k[k]
                r_k = recall_at_k[k]
                f1_at_k[k] = (2 * p_k * r_k) / max(p_k + r_k, 1e-8)
                
                # NDCG@K
                ndcg_at_k[k] = self._calculate_ndcg(
                    retrieved_ids[:k], relevant_ids, highly_relevant_ids
                )
        
        # Mean Reciprocal Rank
        mrr = self._calculate_mrr(retrieved_ids, relevant_ids)
        
        # Mean Average Precision
        map_score = self._calculate_map(retrieved_ids, relevant_ids)
        
        # Coverage (how many relevant docs were found)
        coverage = len(set(retrieved_ids).intersection(relevant_ids)) / max(len(relevant_ids), 1)
        
        # Diversity (how diverse are the retrieved results)
        diversity = self._calculate_diversity(results)
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            mean_reciprocal_rank=mrr,
            ndcg_at_k=ndcg_at_k,
            map_score=map_score,
            coverage=coverage,
            diversity=diversity
        )
    
    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        highly_relevant_ids: Set[str]
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        
        # Assign relevance scores
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in highly_relevant_ids:
                relevance = 2
            elif doc_id in relevant_ids:
                relevance = 1
            else:
                relevance = 0
            
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate ideal DCG
        ideal_scores = [2] * len(highly_relevant_ids) + [1] * (len(relevant_ids) - len(highly_relevant_ids))
        ideal_dcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        return dcg / max(ideal_dcg, 1e-8)
    
    def _calculate_mrr(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_map(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculate Mean Average Precision"""
        if not relevant_ids:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_ids)
    
    def _calculate_diversity(self, results: List[RankedResult]) -> float:
        """Calculate diversity of retrieved results"""
        if len(results) <= 1:
            return 1.0
        
        # Diversity based on data types
        data_types = [result.result.metadata.data_type for result in results]
        unique_types = set(data_types)
        type_diversity = len(unique_types) / len(results)
        
        # Diversity based on source files
        source_files = [result.result.metadata.source_file for result in results if result.result.metadata.source_file]
        if source_files:
            unique_sources = set(source_files)
            source_diversity = len(unique_sources) / len(source_files)
        else:
            source_diversity = 1.0
        
        return (type_diversity + source_diversity) / 2

class GenerationEvaluator:
    """Evaluator for response generation quality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smoothing_function = SmoothingFunction().method1 if NLTK_AVAILABLE else None
    
    def evaluate_generation(
        self,
        generated_response: GeneratedResponse,
        ground_truth: GenerationGroundTruth,
        retrieved_context: List[RankedResult]
    ) -> GenerationMetrics:
        """
        Evaluate generation quality against ground truth
        
        Args:
            generated_response: Generated response from RAG system
            ground_truth: Ground truth reference answer and facts
            retrieved_context: Retrieved context used for generation
        
        Returns:
            Comprehensive generation metrics
        """
        
        generated_text = generated_response.response_text
        reference_text = ground_truth.reference_answer
        
        # BLEU scores
        bleu_scores = self._calculate_bleu_scores(generated_text, reference_text)
        
        # ROUGE scores
        rouge_scores = self._calculate_rouge_scores(generated_text, reference_text)
        
        # Semantic similarity
        semantic_similarity = self._calculate_semantic_similarity(generated_text, reference_text)
        
        # Fact accuracy
        fact_accuracy = self._calculate_fact_accuracy(generated_text, ground_truth.key_facts)
        
        # Entity coverage
        entity_coverage = self._calculate_entity_coverage(
            generated_text, ground_truth.expected_entities or []
        )
        
        # Hallucination score
        hallucination_score = self._calculate_hallucination_score(
            generated_text, retrieved_context
        )
        
        # Confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(
            generated_response.confidence_score, fact_accuracy
        )
        
        return GenerationMetrics(
            bleu_scores=bleu_scores,
            rouge_scores=rouge_scores,
            semantic_similarity=semantic_similarity,
            fact_accuracy=fact_accuracy,
            entity_coverage=entity_coverage,
            hallucination_score=hallucination_score,
            confidence_calibration=confidence_calibration
        )
    
    def _calculate_bleu_scores(self, generated: str, reference: str) -> Dict[int, float]:
        """Calculate BLEU scores for n-grams 1-4"""
        if not NLTK_AVAILABLE:
            return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        
        generated_tokens = generated.lower().split()
        reference_tokens = reference.lower().split()
        
        bleu_scores = {}
        for n in range(1, 5):
            try:
                weights = [1.0/n] * n + [0.0] * (4-n)
                score = sentence_bleu(
                    [reference_tokens],
                    generated_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                bleu_scores[n] = score
            except Exception:
                bleu_scores[n] = 0.0
        
        return bleu_scores
    
    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        # Simple implementation of ROUGE-L (Longest Common Subsequence)
        generated_tokens = generated.lower().split()
        reference_tokens = reference.lower().split()
        
        # ROUGE-1 (unigram overlap)
        gen_unigrams = set(generated_tokens)
        ref_unigrams = set(reference_tokens)
        
        if len(ref_unigrams) == 0:
            rouge_1 = 0.0
        else:
            rouge_1 = len(gen_unigrams.intersection(ref_unigrams)) / len(ref_unigrams)
        
        # ROUGE-L (simplified LCS-based)
        lcs_length = self._longest_common_subsequence_length(generated_tokens, reference_tokens)
        if len(reference_tokens) == 0:
            rouge_l = 0.0
        else:
            rouge_l = lcs_length / len(reference_tokens)
        
        return {
            "rouge_1": rouge_1,
            "rouge_l": rouge_l
        }
    
    def _longest_common_subsequence_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_semantic_similarity(self, generated: str, reference: str) -> float:
        """Calculate semantic similarity (simplified version)"""
        # Simple word overlap similarity
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if len(ref_words) == 0:
            return 0.0
        
        intersection = len(gen_words.intersection(ref_words))
        union = len(gen_words.union(ref_words))
        
        return intersection / max(union, 1)
    
    def _calculate_fact_accuracy(self, generated: str, key_facts: List[str]) -> float:
        """Calculate how many key facts are preserved in the generated text"""
        if not key_facts:
            return 1.0
        
        generated_lower = generated.lower()
        covered_facts = 0
        
        for fact in key_facts:
            # Simple substring matching (can be improved with more sophisticated NLP)
            if fact.lower() in generated_lower:
                covered_facts += 1
        
        return covered_facts / len(key_facts)
    
    def _calculate_entity_coverage(self, generated: str, expected_entities: List[str]) -> float:
        """Calculate how many expected entities are mentioned"""
        if not expected_entities:
            return 1.0
        
        generated_lower = generated.lower()
        covered_entities = 0
        
        for entity in expected_entities:
            if entity.lower() in generated_lower:
                covered_entities += 1
        
        return covered_entities / len(expected_entities)
    
    def _calculate_hallucination_score(
        self,
        generated: str,
        retrieved_context: List[RankedResult]
    ) -> float:
        """Calculate hallucination score (lower is better)"""
        if not retrieved_context:
            return 1.0  # High hallucination if no context
        
        # Extract all text from retrieved context
        context_text = " ".join([result.result.text for result in retrieved_context])
        context_words = set(context_text.lower().split())
        
        generated_words = set(generated.lower().split())
        
        # Common words that don't indicate hallucination
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
        }
        
        # Remove common words from analysis
        content_words = generated_words - common_words
        context_content_words = context_words - common_words
        
        if len(content_words) == 0:
            return 0.0
        
        # Calculate words not supported by context
        unsupported_words = len(content_words - context_content_words)
        hallucination_rate = unsupported_words / len(content_words)
        
        return hallucination_rate
    
    def _calculate_confidence_calibration(
        self,
        predicted_confidence: float,
        actual_accuracy: float
    ) -> float:
        """Calculate how well-calibrated the confidence score is"""
        # Perfect calibration means confidence equals accuracy
        calibration_error = abs(predicted_confidence - actual_accuracy)
        return 1.0 - calibration_error

class RAGSystemEvaluator:
    """Comprehensive RAG system evaluator"""
    
    def __init__(self, rag_system: ForensicRAGSystem, config: Dict[str, Any]):
        self.rag_system = rag_system
        self.config = config
        self.retrieval_evaluator = RetrievalEvaluator(config)
        self.generation_evaluator = GenerationEvaluator(config)
    
    def evaluate_system(
        self,
        test_queries: List[RAGQuery],
        retrieval_ground_truth: List[RetrievalGroundTruth],
        generation_ground_truth: List[GenerationGroundTruth]
    ) -> Dict[str, Any]:
        """
        Comprehensive system evaluation
        
        Args:
            test_queries: List of test queries
            retrieval_ground_truth: Ground truth for retrieval evaluation
            generation_ground_truth: Ground truth for generation evaluation
        
        Returns:
            Complete evaluation results
        """
        
        logger.info(f"Starting comprehensive evaluation with {len(test_queries)} queries")
        start_time = time.time()
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_queries": len(test_queries),
            "retrieval_metrics": [],
            "generation_metrics": [],
            "performance_metrics": {},
            "error_analysis": {"errors": [], "error_rate": 0.0},
            "summary_metrics": {}
        }
        
        response_times = []
        errors = 0
        
        # Create lookup dictionaries
        retrieval_gt_dict = {gt.query: gt for gt in retrieval_ground_truth}
        generation_gt_dict = {gt.query: gt for gt in generation_ground_truth}
        
        for i, query in enumerate(test_queries):
            try:
                logger.info(f"Evaluating query {i+1}/{len(test_queries)}: {query.text[:50]}...")
                
                # Execute query
                query_start_time = time.time()
                response = self.rag_system.query(query)
                query_time = time.time() - query_start_time
                response_times.append(query_time)
                
                # Evaluate retrieval if ground truth available
                if query.text in retrieval_gt_dict:
                    retrieval_gt = retrieval_gt_dict[query.text]
                    retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
                        response.retrieved_results,
                        retrieval_gt
                    )
                    results["retrieval_metrics"].append({
                        "query": query.text,
                        "metrics": asdict(retrieval_metrics)
                    })
                
                # Evaluate generation if ground truth available
                if query.text in generation_gt_dict:
                    generation_gt = generation_gt_dict[query.text]
                    generation_metrics = self.generation_evaluator.evaluate_generation(
                        response.response,
                        generation_gt,
                        response.retrieved_results
                    )
                    results["generation_metrics"].append({
                        "query": query.text,
                        "metrics": asdict(generation_metrics)
                    })
                
            except Exception as e:
                errors += 1
                error_msg = f"Query {i+1} failed: {str(e)}"
                results["error_analysis"]["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Calculate performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        results["performance_metrics"] = {
            "total_evaluation_time": total_time,
            "average_response_time": statistics.mean(response_times) if response_times else 0.0,
            "median_response_time": statistics.median(response_times) if response_times else 0.0,
            "throughput_qps": len(test_queries) / total_time if total_time > 0 else 0.0,
            "successful_queries": len(test_queries) - errors
        }
        
        results["error_analysis"]["error_rate"] = errors / len(test_queries)
        
        # Calculate summary metrics
        results["summary_metrics"] = self._calculate_summary_metrics(results)
        
        logger.info(f"✓ Evaluation completed in {total_time:.2f}s")
        logger.info(f"  - Error rate: {results['error_analysis']['error_rate']:.2%}")
        logger.info(f"  - Average response time: {results['performance_metrics']['average_response_time']:.3f}s")
        
        return results
    
    def _calculate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics from detailed results"""
        summary = {}
        
        # Retrieval summary
        if results["retrieval_metrics"]:
            retrieval_data = [item["metrics"] for item in results["retrieval_metrics"]]
            
            # Average precision, recall, F1 at different k values
            for k in [1, 3, 5, 10]:
                precisions = [data["precision_at_k"].get(str(k), 0.0) for data in retrieval_data]
                recalls = [data["recall_at_k"].get(str(k), 0.0) for data in retrieval_data]
                f1s = [data["f1_at_k"].get(str(k), 0.0) for data in retrieval_data]
                
                if precisions:
                    summary[f"avg_precision_at_{k}"] = statistics.mean(precisions)
                    summary[f"avg_recall_at_{k}"] = statistics.mean(recalls)
                    summary[f"avg_f1_at_{k}"] = statistics.mean(f1s)
            
            # Other retrieval metrics
            mrr_scores = [data["mean_reciprocal_rank"] for data in retrieval_data]
            map_scores = [data["map_score"] for data in retrieval_data]
            
            if mrr_scores:
                summary["avg_mrr"] = statistics.mean(mrr_scores)
            if map_scores:
                summary["avg_map"] = statistics.mean(map_scores)
        
        # Generation summary
        if results["generation_metrics"]:
            generation_data = [item["metrics"] for item in results["generation_metrics"]]
            
            # BLEU scores
            for n in [1, 2, 3, 4]:
                bleu_scores = [data["bleu_scores"].get(str(n), 0.0) for data in generation_data]
                if bleu_scores:
                    summary[f"avg_bleu_{n}"] = statistics.mean(bleu_scores)
            
            # Other generation metrics
            fact_accuracies = [data["fact_accuracy"] for data in generation_data]
            semantic_sims = [data["semantic_similarity"] for data in generation_data]
            hallucination_scores = [data["hallucination_score"] for data in generation_data]
            
            if fact_accuracies:
                summary["avg_fact_accuracy"] = statistics.mean(fact_accuracies)
            if semantic_sims:
                summary["avg_semantic_similarity"] = statistics.mean(semantic_sims)
            if hallucination_scores:
                summary["avg_hallucination_score"] = statistics.mean(hallucination_scores)
        
        return summary
    
    def create_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str
    ):
        """Create a comprehensive evaluation report"""
        
        report = {
            "evaluation_summary": evaluation_results["summary_metrics"],
            "performance_analysis": evaluation_results["performance_metrics"],
            "error_analysis": evaluation_results["error_analysis"],
            "detailed_results": {
                "retrieval_metrics": evaluation_results["retrieval_metrics"],
                "generation_metrics": evaluation_results["generation_metrics"]
            },
            "recommendations": self._generate_recommendations(evaluation_results)
        }
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {output_file}")
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        summary = results.get("summary_metrics", {})
        
        # Retrieval recommendations
        avg_precision_5 = summary.get("avg_precision_at_5", 0.0)
        if avg_precision_5 < 0.5:
            recommendations.append("Consider improving embedding quality or retrieval ranking algorithm")
        
        avg_recall_5 = summary.get("avg_recall_at_5", 0.0)
        if avg_recall_5 < 0.3:
            recommendations.append("Consider expanding the knowledge base or improving query analysis")
        
        # Generation recommendations
        avg_fact_accuracy = summary.get("avg_fact_accuracy", 0.0)
        if avg_fact_accuracy < 0.7:
            recommendations.append("Consider improving fact extraction and verification in generation")
        
        avg_hallucination = summary.get("avg_hallucination_score", 0.0)
        if avg_hallucination > 0.3:
            recommendations.append("Consider adding stricter grounding constraints to reduce hallucinations")
        
        # Performance recommendations
        avg_response_time = results.get("performance_metrics", {}).get("average_response_time", 0.0)
        if avg_response_time > 5.0:
            recommendations.append("Consider optimizing vector search or using faster embedding models")
        
        error_rate = results.get("error_analysis", {}).get("error_rate", 0.0)
        if error_rate > 0.1:
            recommendations.append("Investigate and fix system reliability issues")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges")
        
        return recommendations

# Utility functions for creating test data
def create_sample_ground_truth() -> Tuple[List[RetrievalGroundTruth], List[GenerationGroundTruth]]:
    """Create sample ground truth data for testing"""
    
    retrieval_ground_truth = [
        RetrievalGroundTruth(
            query="What crypto transactions occurred?",
            relevant_document_ids={"msg_002", "msg_003", "comm_001"},
            highly_relevant_ids={"msg_002", "msg_003"},
            query_type="entity",
            expected_data_types=["conversation", "entity"]
        ),
        RetrievalGroundTruth(
            query="Who contacted Alex Rivera?",
            relevant_document_ids={"contact_001", "call_001", "msg_001"},
            highly_relevant_ids={"contact_001"},
            query_type="entity",
            expected_data_types=["contact", "call_log", "conversation"]
        )
    ]
    
    generation_ground_truth = [
        GenerationGroundTruth(
            query="What crypto transactions occurred?",
            reference_answer="A cryptocurrency transaction of 50,000 was mentioned, with funds being sent to Bitcoin address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.",
            key_facts=["50k crypto", "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "transaction complete"],
            expected_entities=["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"],
            answer_type="factual"
        ),
        GenerationGroundTruth(
            query="Who contacted Alex Rivera?",
            reference_answer="Based on the communication records, the device owner contacted Alex Rivera at phone number +1-555-0987.",
            key_facts=["+1-555-0987", "Alex Rivera", "contact"],
            expected_entities=["Alex Rivera", "+1-555-0987"],
            answer_type="factual"
        )
    ]
    
    return retrieval_ground_truth, generation_ground_truth