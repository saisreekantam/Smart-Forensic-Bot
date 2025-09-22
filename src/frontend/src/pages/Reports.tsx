import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  FileText, 
  Download, 
  MessageSquare, 
  Clock, 
  Users, 
  BarChart3,
  Loader2,
  AlertTriangle
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface Report {
  id: string;
  title: string;
  description: string;
  type: string;
  available: boolean;
}

interface ReportData {
  title: string;
  case_id: string;
  generated_at: string;
  [key: string]: any;
}

const Reports = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [generatingReport, setGeneratingReport] = useState<string | null>(null);
  const [generatedReports, setGeneratedReports] = useState<Record<string, ReportData>>({});
  const { toast } = useToast();

  useEffect(() => {
    loadAvailableReports();
  }, []);

  const loadAvailableReports = async () => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      toast({
        title: "No case selected",
        description: "Please select a case first.",
        variant: "destructive",
      });
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/reports`);
      if (!response.ok) {
        throw new Error('Failed to load reports');
      }
      
      const data = await response.json();
      setReports(data.reports || []);
    } catch (error) {
      console.error('Error loading reports:', error);
      toast({
        title: "Error loading reports",
        description: "Failed to load available reports.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async (reportId: string) => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) return;

    setGeneratingReport(reportId);

    try {
      const response = await fetch(
        `http://localhost:8000/cases/${selectedCaseId}/reports/${reportId}/generate`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const reportData = await response.json();
      setGeneratedReports(prev => ({
        ...prev,
        [reportId]: reportData.data
      }));

      toast({
        title: "Report generated",
        description: `${reportData.data.title} has been generated successfully.`,
      });
    } catch (error) {
      console.error('Error generating report:', error);
      toast({
        title: "Error generating report",
        description: "Failed to generate the requested report.",
        variant: "destructive",
      });
    } finally {
      setGeneratingReport(null);
    }
  };

  const getReportIcon = (type: string) => {
    switch (type) {
      case 'summary':
        return <FileText className="h-5 w-5" />;
      case 'analysis':
        return <MessageSquare className="h-5 w-5" />;
      case 'timeline':
        return <Clock className="h-5 w-5" />;
      case 'network':
        return <Users className="h-5 w-5" />;
      default:
        return <BarChart3 className="h-5 w-5" />;
    }
  };

  const getReportColor = (type: string) => {
    switch (type) {
      case 'summary':
        return 'bg-blue-100 text-blue-800';
      case 'analysis':
        return 'bg-green-100 text-green-800';
      case 'timeline':
        return 'bg-purple-100 text-purple-800';
      case 'network':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const exportReport = (reportId: string, reportData: ReportData) => {
    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${reportId}_${reportData.case_id}_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin" />
          <span className="ml-2">Loading reports...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Investigation Reports</h1>
        <Badge variant="outline">
          Case: {localStorage.getItem('selectedCaseId') || 'None'}
        </Badge>
      </div>

      {reports.length === 0 ? (
        <Card>
          <CardContent className="py-12">
            <div className="text-center">
              <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-orange-500" />
              <h3 className="text-lg font-semibold mb-2">No Reports Available</h3>
              <p className="text-muted-foreground">
                Process some evidence data first to generate reports.
              </p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {reports.map((report) => (
            <Card key={report.id} className={!report.available ? "opacity-60" : ""}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getReportIcon(report.type)}
                    <span>{report.title}</span>
                  </div>
                  <Badge className={getReportColor(report.type)}>
                    {report.type.toUpperCase()}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground">{report.description}</p>
                
                {!report.available ? (
                  <div className="flex items-center gap-2 text-sm text-orange-600">
                    <AlertTriangle className="h-4 w-4" />
                    <span>Not available - missing required data</span>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Button
                      onClick={() => generateReport(report.id)}
                      disabled={generatingReport === report.id || !report.available}
                      className="w-full"
                    >
                      {generatingReport === report.id ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <BarChart3 className="h-4 w-4 mr-2" />
                          Generate Report
                        </>
                      )}
                    </Button>

                    {generatedReports[report.id] && (
                      <>
                        <Separator />
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">
                              Report Generated
                            </span>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => exportReport(report.id, generatedReports[report.id])}
                            >
                              <Download className="h-4 w-4 mr-2" />
                              Export
                            </Button>
                          </div>
                          
                          <div className="bg-muted p-3 rounded-lg">
                            <div className="text-sm space-y-1">
                              <div>
                                <strong>Generated:</strong>{" "}
                                {new Date(generatedReports[report.id].generated_at).toLocaleString()}
                              </div>
                              <div>
                                <strong>Case ID:</strong>{" "}
                                {generatedReports[report.id].case_id}
                              </div>
                              {generatedReports[report.id].summary && (
                                <div>
                                  <strong>Summary:</strong>{" "}
                                  {generatedReports[report.id].summary}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Report Types Guide */}
      <Card>
        <CardHeader>
          <CardTitle>Report Types Guide</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <FileText className="h-5 w-5 text-blue-600" />
                <div>
                  <h4 className="font-semibold">Evidence Summary</h4>
                  <p className="text-sm text-muted-foreground">
                    Overview of all evidence items and data sources
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <MessageSquare className="h-5 w-5 text-green-600" />
                <div>
                  <h4 className="font-semibold">Communication Analysis</h4>
                  <p className="text-sm text-muted-foreground">
                    Analysis of calls, messages, and communication patterns
                  </p>
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <Clock className="h-5 w-5 text-purple-600" />
                <div>
                  <h4 className="font-semibold">Timeline Report</h4>
                  <p className="text-sm text-muted-foreground">
                    Chronological timeline of events and activities
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <Users className="h-5 w-5 text-orange-600" />
                <div>
                  <h4 className="font-semibold">Contact Network</h4>
                  <p className="text-sm text-muted-foreground">
                    Network analysis of contacts and relationships
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Reports;