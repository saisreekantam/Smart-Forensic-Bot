/**
 * Evidence Service - Handles all evidence-related API calls
 */

import { apiService, ApiError } from './api';

// Types matching the backend API
export interface Evidence {
  id: string;
  original_filename: string;
  evidence_type: string;
  processing_status: string;
  file_size: number;
  created_at: string;
  has_embeddings: boolean;
}

export interface EvidenceUploadRequest {
  evidence_type: string;
  title?: string;
  description?: string;
  source_device?: string;
  extraction_method?: string;
}

export interface TimelineEvent {
  timestamp: string;
  event: string;
  evidence: string[];
  importance: 'low' | 'medium' | 'high' | 'critical';
}

export interface EvidenceTimelineResponse {
  timeline_events: TimelineEvent[];
  insights: string[];
  confidence: number;
  generated_by: string;
  case_id: string;
}

// Enhanced Evidence interface for frontend (combining API data with frontend needs)
export interface EnhancedEvidence extends Evidence {
  name: string; // Maps to original_filename
  type: 'document' | 'image' | 'audio' | 'video' | 'call_log' | 'message' | 'email';
  size: string; // Formatted file size
  timestamp: string; // Maps to created_at
  source: string; // Derived from evidence_type or metadata
  aiConfidence: number; // Mock value for now, could come from AI analysis
  entities: string[]; // Mock values for now
  relationships: number; // Mock value for now
  summary: string; // Mock value for now, could come from AI analysis
  metadata: Record<string, any>; // Additional metadata
  tags: string[]; // Mock values for now
}

export class EvidenceService {
  /**
   * Get all evidence for a case
   */
  async getCaseEvidence(caseId: string, evidenceType?: string): Promise<Evidence[]> {
    try {
      const params = evidenceType ? `?evidence_type=${encodeURIComponent(evidenceType)}` : '';
      return await apiService.get<Evidence[]>(`/cases/${caseId}/evidence${params}`);
    } catch (error) {
      console.error('Failed to fetch case evidence:', error);
      throw error;
    }
  }

  /**
   * Upload evidence file to a case
   */
  async uploadEvidence(
    caseId: string,
    file: File,
    request: EvidenceUploadRequest
  ): Promise<{ message: string; evidence_id: string; filename: string; status: string; processing: string }> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('evidence_type', request.evidence_type);
      
      if (request.title) formData.append('title', request.title);
      if (request.description) formData.append('description', request.description);
      if (request.source_device) formData.append('source_device', request.source_device);
      if (request.extraction_method) formData.append('extraction_method', request.extraction_method);

      return await apiService.uploadFile(`/cases/${caseId}/evidence`, formData);
    } catch (error) {
      console.error('Failed to upload evidence:', error);
      throw error;
    }
  }

  /**
   * Simple evidence upload (uses the simpler endpoint)
   */
  async uploadEvidenceSimple(
    caseId: string,
    file: File,
    description?: string
  ): Promise<{ message: string; evidence_id: string; filename: string; status: string }> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (description) formData.append('description', description);

      return await apiService.uploadFile(`/cases/${caseId}/evidence/upload`, formData);
    } catch (error) {
      console.error('Failed to upload evidence (simple):', error);
      throw error;
    }
  }

  /**
   * Get evidence timeline data
   */
  async getEvidenceTimeline(caseId: string): Promise<EvidenceTimelineResponse> {
    try {
      return await apiService.get<EvidenceTimelineResponse>(`/cases/${caseId}/evidence/timeline`);
    } catch (error) {
      console.error('Failed to fetch evidence timeline:', error);
      throw error;
    }
  }

  /**
   * Transform API Evidence to Enhanced Evidence for frontend use
   */
  transformToEnhancedEvidence(evidence: Evidence): EnhancedEvidence {
    // Map evidence types
    const typeMapping: Record<string, EnhancedEvidence['type']> = {
      'document': 'document',
      'image': 'image', 
      'audio': 'audio',
      'video': 'video',
      'call_log': 'call_log',
      'message': 'message',
      'email': 'email',
      'text': 'document',
      'pdf': 'document',
      'phone_data': 'call_log'
    };

    // Format file size
    const formatFileSize = (bytes: number): string => {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    };

    // Generate mock AI confidence based on processing status
    const aiConfidence = evidence.processing_status === 'completed' ? 
      0.85 + Math.random() * 0.1 : // 85-95% for completed
      0.5 + Math.random() * 0.3;   // 50-80% for others

    // Generate mock entities based on evidence type
    const generateMockEntities = (type: string, filename: string): string[] => {
      const entities: string[] = [];
      if (type.includes('phone') || type.includes('call')) {
        entities.push('+1234567890', 'John Doe', 'Contact Name');
      }
      if (type.includes('email') || type.includes('message')) {
        entities.push('user@email.com', 'John Smith', 'Message Thread');
      }
      if (type.includes('document')) {
        entities.push('Document Entity', 'Key Person', 'Important Date');
      }
      return entities.slice(0, 2 + Math.floor(Math.random() * 3)); // 2-4 entities
    };

    // Generate mock summary
    const generateSummary = (type: string, filename: string): string => {
      const summaries = {
        document: `Document analysis reveals key information relevant to the investigation. AI processing identified important entities and relationships.`,
        image: `Image analysis completed with facial recognition and object detection. ${filename} contains relevant visual evidence.`,
        audio: `Audio file processed with speech-to-text analysis. Key conversations and speakers identified in ${filename}.`,
        video: `Video analysis completed with scene detection and person tracking. Important activities captured in ${filename}.`,
        call_log: `Call log data processed showing communication patterns. Analysis of ${filename} reveals contact networks.`,
        message: `Message analysis completed with sentiment and entity extraction. ${filename} contains important communications.`,
        email: `Email analysis reveals correspondence patterns and key participants. ${filename} processed for investigative insights.`
      };
      return summaries[typeMapping[evidence.evidence_type] || 'document'] || 
             `Evidence file ${filename} has been processed and analyzed for investigative insights.`;
    };

    // Generate mock tags
    const generateTags = (type: string): string[] => {
      const tagSets = {
        document: ['document', 'text-analysis', 'entities'],
        image: ['visual-evidence', 'image-analysis', 'metadata'],
        audio: ['audio-evidence', 'speech-to-text', 'conversations'],
        video: ['video-evidence', 'visual-analysis', 'surveillance'],
        call_log: ['communications', 'call-data', 'network-analysis'],
        message: ['messages', 'communications', 'text-analysis'],
        email: ['email', 'communications', 'correspondence']
      };
      
      const baseTags = tagSets[typeMapping[evidence.evidence_type] || 'document'] || ['evidence'];
      const priorityTags = evidence.processing_status === 'completed' ? ['processed'] : ['pending'];
      const aiTags = aiConfidence > 0.8 ? ['high-confidence'] : ['medium-confidence'];
      
      return [...baseTags, ...priorityTags, ...aiTags].slice(0, 4);
    };

    const mappedType = typeMapping[evidence.evidence_type] || 'document';
    const entities = generateMockEntities(evidence.evidence_type, evidence.original_filename);

    return {
      ...evidence,
      name: evidence.original_filename,
      type: mappedType,
      size: formatFileSize(evidence.file_size),
      timestamp: evidence.created_at,
      source: evidence.evidence_type.charAt(0).toUpperCase() + evidence.evidence_type.slice(1).replace('_', ' '),
      aiConfidence,
      entities,
      relationships: Math.floor(Math.random() * 15) + 1, // 1-15 relationships
      summary: generateSummary(evidence.evidence_type, evidence.original_filename),
      metadata: {
        processing_status: evidence.processing_status,
        has_embeddings: evidence.has_embeddings,
        file_size_bytes: evidence.file_size
      },
      tags: generateTags(evidence.evidence_type)
    };
  }

  /**
   * Get enhanced evidence for frontend use
   */
  async getEnhancedCaseEvidence(caseId: string, evidenceType?: string): Promise<EnhancedEvidence[]> {
    try {
      const evidence = await this.getCaseEvidence(caseId, evidenceType);
      return evidence.map(item => this.transformToEnhancedEvidence(item));
    } catch (error) {
      console.error('Failed to fetch enhanced case evidence:', error);
      throw error;
    }
  }
}

export const evidenceService = new EvidenceService();