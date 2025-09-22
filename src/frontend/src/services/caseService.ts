/**
 * Case Service - Handles all case-related API calls
 */

import { apiService } from './api';

export interface Case {
  id: string;
  case_number: string;
  title: string;
  status: string;
  investigator_name: string;
  created_at: string;
  updated_at: string;
  total_evidence_count: number;
  processed_evidence_count: number;
  processing_progress: number;
}

export interface CaseCreateRequest {
  case_number: string;
  title: string;
  investigator_name: string;
  description?: string;
  investigator_id?: string;
  department?: string;
  incident_date?: string;
  due_date?: string;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  case_type?: string;
  jurisdiction?: string;
  tags?: string[];
}

export class CaseService {
  /**
   * Get all cases
   */
  async getCases(): Promise<Case[]> {
    try {
      return await apiService.get<Case[]>('/cases');
    } catch (error) {
      console.error('Failed to fetch cases:', error);
      throw error;
    }
  }

  /**
   * Resolve case number to case ID
   */
  async resolveCaseId(caseNumberOrId: string): Promise<string> {
    try {
      // First try to get the case directly (in case it's already an ID)
      try {
        await this.getCase(caseNumberOrId);
        return caseNumberOrId; // It's already a valid ID
      } catch (error) {
        // If that fails, it might be a case number, so fetch all cases and find by number
        const cases = await this.getCases();
        const caseByNumber = cases.find(c => c.case_number === caseNumberOrId);
        
        if (caseByNumber) {
          return caseByNumber.id;
        }
        
        throw new Error(`Case not found: ${caseNumberOrId}`);
      }
    } catch (error) {
      console.error(`Failed to resolve case ID for ${caseNumberOrId}:`, error);
      throw error;
    }
  }

  /**
   * Get a specific case by ID
   */
  async getCase(caseId: string): Promise<Case> {
    try {
      return await apiService.get<Case>(`/cases/${caseId}`);
    } catch (error) {
      console.error(`Failed to fetch case ${caseId}:`, error);
      throw error;
    }
  }

  /**
   * Create a new case
   */
  async createCase(caseData: CaseCreateRequest): Promise<Case> {
    try {
      return await apiService.post<Case>('/cases', caseData);
    } catch (error) {
      console.error('Failed to create case:', error);
      throw error;
    }
  }

  /**
   * Process case data
   */
  async processCase(caseId: string): Promise<{ message: string; status: string; results?: any }> {
    try {
      return await apiService.post(`/cases/${caseId}/process`);
    } catch (error) {
      console.error(`Failed to process case ${caseId}:`, error);
      throw error;
    }
  }

  /**
   * Get case processing status
   */
  async getCaseStatus(caseId: string): Promise<{ status: string; progress: number; message?: string }> {
    try {
      return await apiService.get(`/cases/${caseId}/status`);
    } catch (error) {
      console.error(`Failed to get case status ${caseId}:`, error);
      throw error;
    }
  }
}

export const caseService = new CaseService();