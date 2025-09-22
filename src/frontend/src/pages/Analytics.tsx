import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  BarChart3, 
  Users, 
  MessageSquare, 
  Clock, 
  TrendingUp,
  Database,
  Loader2,
  AlertTriangle,
  Phone,
  Mail,
  FileText
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface AnalyticsData {
  total_records: number;
  data_sources: Array<{
    file: string;
    records: number;
  }>;
  communication_stats: {
    total_communications: number;
    by_type: Record<string, number>;
    unique_contacts: number;
  };
  timeline_data: Array<{
    timestamp: string;
    event: string;
  }>;
  contact_network: Array<{
    name: string;
    frequency: number;
  }>;
}

const Analytics = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      setError("No case selected");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/analytics`);
      if (!response.ok) {
        throw new Error('Failed to load analytics');
      }
      
      const data = await response.json();
      setAnalyticsData(data.analytics);
      setError(null);
    } catch (error) {
      console.error('Error loading analytics:', error);
      setError("Failed to load analytics data");
      toast({
        title: "Error loading analytics",
        description: "Failed to load analytics data.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  const getDataSourceIcon = (filename: string) => {
    const lower = filename.toLowerCase();
    if (lower.includes('call') || lower.includes('phone')) {
      return <Phone className="h-4 w-4" />;
    } else if (lower.includes('message') || lower.includes('sms')) {
      return <Mail className="h-4 w-4" />;
    } else {
      return <FileText className="h-4 w-4" />;
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin" />
          <span className="ml-2">Loading analytics...</span>
        </div>
      </div>
    );
  }

  if (error || !analyticsData) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="py-12">
            <div className="text-center">
              <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-orange-500" />
              <h3 className="text-lg font-semibold mb-2">
                {error || "No Analytics Data Available"}
              </h3>
              <p className="text-muted-foreground">
                {error === "No case selected" 
                  ? "Please select a case first."
                  : "Process some evidence data first to view analytics."}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Data Analytics</h1>
        <Badge variant="outline">
          Case: {localStorage.getItem('selectedCaseId') || 'None'}
        </Badge>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Records</p>
                <p className="text-2xl font-bold">{formatNumber(analyticsData.total_records)}</p>
              </div>
              <Database className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Data Sources</p>
                <p className="text-2xl font-bold">{analyticsData.data_sources.length}</p>
              </div>
              <FileText className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Communications</p>
                <p className="text-2xl font-bold">
                  {formatNumber(analyticsData.communication_stats.total_communications)}
                </p>
              </div>
              <MessageSquare className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Unique Contacts</p>
                <p className="text-2xl font-bold">
                  {formatNumber(analyticsData.communication_stats.unique_contacts)}
                </p>
              </div>
              <Users className="h-8 w-8 text-orange-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Sources */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Data Sources
            </CardTitle>
          </CardHeader>
          <CardContent>
            {analyticsData.data_sources.length === 0 ? (
              <p className="text-muted-foreground text-center py-4">No data sources found</p>
            ) : (
              <div className="space-y-3">
                {analyticsData.data_sources.map((source, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      {getDataSourceIcon(source.file)}
                      <div>
                        <p className="font-medium">{source.file}</p>
                        <p className="text-sm text-muted-foreground">
                          {typeof source.records === 'number' ? formatNumber(source.records) : source.records} records
                        </p>
                      </div>
                    </div>
                    <Badge variant="outline">
                      {source.file.split('.').pop()?.toUpperCase()}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Communication Types */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5" />
              Communication Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            {Object.keys(analyticsData.communication_stats.by_type).length === 0 ? (
              <p className="text-muted-foreground text-center py-4">No communication data found</p>
            ) : (
              <div className="space-y-3">
                {Object.entries(analyticsData.communication_stats.by_type).map(([type, count]) => {
                  const total = analyticsData.communication_stats.total_communications;
                  const percentage = total > 0 ? (count / total) * 100 : 0;
                  
                  return (
                    <div key={type} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium capitalize">{type}</span>
                        <span>{formatNumber(count)} ({percentage.toFixed(1)}%)</span>
                      </div>
                      <Progress value={percentage} className="h-2" />
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Top Contacts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Top Contacts
            </CardTitle>
          </CardHeader>
          <CardContent>
            {analyticsData.contact_network.length === 0 ? (
              <p className="text-muted-foreground text-center py-4">No contact data found</p>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {analyticsData.contact_network.slice(0, 10).map((contact, index) => (
                  <div key={index} className="flex items-center justify-between p-2 border rounded">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
                        {contact.name.charAt(0).toUpperCase()}
                      </div>
                      <span className="font-medium">{contact.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">{formatNumber(contact.frequency)}</Badge>
                      <TrendingUp className="h-4 w-4 text-green-600" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Timeline Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            {analyticsData.timeline_data.length === 0 ? (
              <p className="text-muted-foreground text-center py-4">No timeline data found</p>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {analyticsData.timeline_data.slice(0, 8).map((event, index) => (
                  <div key={index} className="flex items-center gap-3 p-2 border-l-2 border-blue-200 pl-4">
                    <div className="flex-1">
                      <p className="font-medium text-sm">{event.event}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(event.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                ))}
                {analyticsData.timeline_data.length > 8 && (
                  <div className="text-center pt-2 border-t">
                    <Badge variant="outline">
                      +{analyticsData.timeline_data.length - 8} more events
                    </Badge>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Summary Card */}
      <Card>
        <CardHeader>
          <CardTitle>Analytics Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {formatNumber(analyticsData.total_records)}
              </div>
              <p className="text-sm text-muted-foreground">
                Total evidence records processed across all data sources
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {analyticsData.data_sources.length}
              </div>
              <p className="text-sm text-muted-foreground">
                Different data sources contributing to the investigation
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">
                {formatNumber(analyticsData.communication_stats.unique_contacts)}
              </div>
              <p className="text-sm text-muted-foreground">
                Unique contacts identified in communication records
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Analytics;