import { 
  FileText, 
  Search, 
  Upload, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Users,
  Database,
  TrendingUp,
  Play
} from "lucide-react";
import { useState, useEffect } from "react";
import { StatsCard } from "@/components/dashboard/StatsCard";
import { RecentActivity } from "@/components/dashboard/RecentActivity";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";


interface CaseData {
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
  description?: string;
}

interface Evidence {
  id: string;
  original_filename: string;
  evidence_type: string;
  processing_status: string;
  file_size: number;
  created_at: string;
  has_embeddings: boolean;
}

const Dashboard = () => {
  const [caseData, setCaseData] = useState<CaseData | null>(null);
  const [evidenceList, setEvidenceList] = useState<Evidence[]>([]);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [uploadMessage, setUploadMessage] = useState<string>("");
  const [processingStatus, setProcessingStatus] = useState<"idle" | "processing" | "success" | "error">("idle");
  const [processingMessage, setProcessingMessage] = useState<string>("");
  const [loading, setLoading] = useState(true);

  const { toast } = useToast();
  const navigate = useNavigate();

  useEffect(() => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      navigate('/');
      return;
    }
    fetchCaseData(selectedCaseId);
    fetchEvidenceData(selectedCaseId);
  }, [navigate]);

  const fetchCaseData = async (caseId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/cases/${caseId}`);
      if (response.ok) {
        const data = await response.json();
        setCaseData(data.case);
      } else {
        toast({
          title: "Error",
          description: "Failed to load case data",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('Error fetching case data:', error);
      toast({
        title: "Error",
        description: "Failed to connect to backend",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchEvidenceData = async (caseId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/cases/${caseId}/evidence`);
      if (response.ok) {
        const data = await response.json();
        setEvidenceList(data);
      }
    } catch (error) {
      console.error('Error fetching evidence data:', error);
    }
  };

  const handleUpload = async (file: File) => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      toast({
        title: "Error",
        description: "No case selected",
        variant: "destructive"
      });
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("evidence_type", "ufdr"); // Default to UFDR
    formData.append("title", file.name);
    formData.append("description", "Evidence uploaded via dashboard");

    setUploadStatus("uploading");
    setUploadMessage("Uploading file...");

    try {
      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/evidence`, {
        method: "POST",
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        setUploadStatus("success");
        setUploadMessage("File uploaded successfully and processing started!");
        // Refresh case and evidence data
        fetchCaseData(selectedCaseId);
        fetchEvidenceData(selectedCaseId);
        toast({
          title: "Success",
          description: "Evidence uploaded and processing started"
        });
      } else {
        const error = await response.json();
        setUploadStatus("error");
        setUploadMessage(error.detail || "Upload failed. Please try again.");
      }
    } catch (error) {
      setUploadStatus("error");
      setUploadMessage("Error uploading file. Please check your connection.");
    }
  };

  const handleProcessData = async () => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      toast({
        title: "Error",
        description: "No case selected",
        variant: "destructive"
      });
      return;
    }

    setProcessingStatus("processing");
    setProcessingMessage("Processing case data...");

    try {
      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      
      if (response.ok) {
        const result = await response.json();
        setProcessingStatus("success");
        
        if (result.status === "up_to_date") {
          setProcessingMessage("All evidence is already processed and ready for analysis!");
        } else {
          setProcessingMessage(
            `Started processing ${result.processing_count} evidence files. ` +
            `Estimated completion: ${result.estimated_completion_minutes} minutes.`
          );
        }
        
        // Refresh case and evidence data
        fetchCaseData(selectedCaseId);
        fetchEvidenceData(selectedCaseId);
        
        toast({
          title: "Success",
          description: result.message
        });
      } else {
        const error = await response.json();
        setProcessingStatus("error");
        setProcessingMessage(error.detail || "Processing failed. Please try again.");
        toast({
          title: "Error",
          description: error.detail || "Processing failed",
          variant: "destructive"
        });
      }
    } catch (error) {
      setProcessingStatus("error");
      setProcessingMessage("Error starting data processing. Please check your connection.");
      toast({
        title: "Error",
        description: "Failed to connect to backend",
        variant: "destructive"
      });
    }
  };

  const calculateTotalSize = () => {
    return evidenceList.reduce((total, evidence) => total + evidence.file_size, 0);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getProcessedCount = () => {
    return evidenceList.filter(evidence => evidence.processing_status === 'completed').length;
  };

  const getQueriesExecuted = () => {
    // This would come from actual query logs in a real implementation
    return evidenceList.filter(evidence => evidence.has_embeddings).length * 10; // Estimate
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!caseData) {
    return (
      <div className="text-center py-20">
        <h2 className="text-2xl font-semibold mb-4">Case not found</h2>
        <Button onClick={() => navigate('/')}>Back to Cases</Button>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-subtle">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 to-slate-900/20" />
        <div className="relative p-8">
          <div className="max-w-2xl">
            <div className="flex items-center gap-3 mb-4">
              <h1 className="text-4xl font-bold">
                {caseData.case_number}
              </h1>
              <Badge variant={caseData.status === 'active' ? 'default' : 'secondary'}>
                {caseData.status}
              </Badge>
            </div>
            <h2 className="text-2xl font-semibold mb-2">{caseData.title}</h2>
            <p className="text-xl text-muted-foreground mb-4">
              Lead Investigator: {caseData.investigator_name}
            </p>
            {caseData.description && (
              <p className="text-lg text-muted-foreground mb-6">
                {caseData.description}
              </p>
            )}
            <div className="flex gap-4">
              <Button variant="professional" size="lg" asChild>
                <label className="cursor-pointer">
                  <Upload className="mr-2 w-5 h-5" />
                  {uploadStatus === "uploading" ? "Uploading..." : "Upload Evidence"}
                  <input
                    type="file"
                    className="hidden"
                    disabled={uploadStatus === "uploading"}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        handleUpload(file);
                      }
                    }}
                  />
                </label>
              </Button>
              <Button 
                variant="investigation" 
                size="lg" 
                onClick={handleProcessData}
                disabled={processingStatus === "processing" || caseData.total_evidence_count === 0}
              >
                <Play className="mr-2 w-5 h-5" />
                {processingStatus === "processing" ? "Processing..." : "Process Data"}
              </Button>
              <Button variant="investigation" size="lg" onClick={() => navigate('/query')}>
                <Search className="mr-2 w-5 h-5" />
                Start Investigation Query
              </Button>
            </div>
            {/* Upload Status Messages */}
            {uploadStatus === "uploading" && (
              <div className="mt-4 text-blue-600 font-medium flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                {uploadMessage}
              </div>
            )}
            {uploadStatus === "success" && (
              <div className="mt-4 text-green-600 font-medium">{uploadMessage}</div>
            )}
            {uploadStatus === "error" && (
              <div className="mt-4 text-red-600 font-medium">{uploadMessage}</div>
            )}
            
            {/* Processing Status Messages */}
            {processingStatus === "processing" && (
              <div className="mt-4 text-blue-600 font-medium flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                {processingMessage}
              </div>
            )}
            {processingStatus === "success" && (
              <div className="mt-4 text-green-600 font-medium">{processingMessage}</div>
            )}
            {processingStatus === "error" && (
              <div className="mt-4 text-red-600 font-medium">{processingMessage}</div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Evidence Files"
          value={caseData.total_evidence_count}
          description="Total files uploaded"
          icon={FileText}
          trend={{ 
            value: caseData.total_evidence_count > 0 ? Math.round(caseData.processing_progress) : 0, 
            label: "processed" 
          }}
        />
        <StatsCard
          title="Evidence Processed"
          value={formatFileSize(calculateTotalSize())}
          description="Total data analyzed"
          icon={Database}
          trend={{ 
            value: getProcessedCount(), 
            label: "files ready" 
          }}
        />
        <StatsCard
          title="Queries Available"
          value={getQueriesExecuted()}
          description="Natural language searches possible"
          icon={Search}
          trend={{ 
            value: evidenceList.filter(e => e.has_embeddings).length, 
            label: "indexed files" 
          }}
        />
        <StatsCard
          title="Processing Progress"
          value={`${Math.round(caseData.processing_progress)}%`}
          description="Evidence analysis completion"
          icon={TrendingUp}
          trend={{ 
            value: caseData.processed_evidence_count, 
            label: `of ${caseData.total_evidence_count} files` 
          }}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Query Interface */}
        <div className="lg:col-span-2">
          <Card className="card-professional">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="w-5 h-5 text-blue-400" />
                AI Investigation Assistant
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Search className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-2">
                  Start AI-Powered Investigation
                </h3>
                <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                  Interact with our advanced AI assistant to analyze evidence, ask questions, and get investigative insights from your case data.
                </p>
                <Button 
                  onClick={() => navigate('/query')}
                  size="lg"
                  className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-8 py-3"
                  disabled={!caseData || caseData.processed_evidence_count === 0}
                >
                  <Play className="w-5 h-5 mr-2" />
                  Launch Investigation Assistant
                </Button>
                {caseData && caseData.processed_evidence_count === 0 && (
                  <p className="text-amber-400 text-sm mt-3 flex items-center justify-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    Upload and process evidence first to enable AI queries
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions & Status */}
        <div className="space-y-6">
          {/* System Status */}
          <Card className="card-professional">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-success" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">AI Processing Engine</span>
                <Badge className="status-active">Online</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Database Connection</span>
                <Badge className="status-active">Connected</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Evidence Storage</span>
                <Badge className="status-active">Available</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Report Generator</span>
                <Badge className={caseData.processing_progress === 100 ? "status-active" : "status-pending"}>
                  {caseData.processing_progress === 100 ? "Ready" : "Processing"}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="card-professional">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button variant="outline" className="w-full justify-start" size="sm" asChild>
                <label className="cursor-pointer">
                  <Upload className="mr-2 w-4 h-4" />
                  Upload New Evidence
                  <input
                    type="file"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleUpload(file);
                    }}
                  />
                </label>
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start" 
                size="sm"
                onClick={handleProcessData}
                disabled={processingStatus === "processing" || caseData.total_evidence_count === 0}
              >
                <Play className="mr-2 w-4 h-4" />
                {processingStatus === "processing" ? "Processing..." : "Process Evidence Data"}
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start" 
                size="sm"
                disabled={caseData.processing_progress < 100}
              >
                <FileText className="mr-2 w-4 h-4" />
                Generate Report
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start" 
                size="sm"
                onClick={() => navigate('/')}
              >
                <Users className="mr-2 w-4 h-4" />
                Case Management
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start" 
                size="sm"
                onClick={() => navigate('/query')}
                disabled={caseData.processed_evidence_count === 0}
              >
                <Search className="mr-2 w-4 h-4" />
                Investigation Query
              </Button>
            </CardContent>
          </Card>

          {/* Processing Status */}
          {caseData.processing_progress < 100 && caseData.total_evidence_count > 0 && (
            <Card className="card-professional border-warning/30">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-warning">
                  <Clock className="w-5 h-5" />
                  Processing Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-3 bg-warning/10 rounded-lg border border-warning/20">
                  <p className="text-sm font-medium text-warning">Evidence Processing in Progress</p>
                  <p className="text-xs text-muted-foreground">
                    {caseData.processed_evidence_count} of {caseData.total_evidence_count} files processed
                  </p>
                  <div className="w-full bg-warning/20 rounded-full h-2 mt-2">
                    <div 
                      className="bg-warning h-2 rounded-full transition-all"
                      style={{ width: `${caseData.processing_progress}%` }}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Evidence Summary */}
          <Card className="card-professional">
            <CardHeader>
              <CardTitle>Evidence Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {evidenceList.length === 0 ? (
                <p className="text-sm text-muted-foreground">No evidence files uploaded yet</p>
              ) : (
                evidenceList.slice(0, 3).map((evidence) => (
                  <div key={evidence.id} className="flex items-center justify-between text-sm">
                    <span className="truncate flex-1">{evidence.original_filename}</span>
                    <Badge 
                      variant={evidence.processing_status === 'completed' ? 'default' : 'secondary'}
                      className="text-xs"
                    >
                      {evidence.processing_status}
                    </Badge>
                  </div>
                ))
              )}
              {evidenceList.length > 3 && (
                <p className="text-xs text-muted-foreground">
                  +{evidenceList.length - 3} more files
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Recent Activity */}
      <RecentActivity />
    </div>
  );
};

export default Dashboard;
