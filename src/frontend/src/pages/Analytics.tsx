import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
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
  FileText,
  PhoneCall,
  PhoneIncoming,
  PhoneOutgoing,
  Timer,
  Calendar,
  Activity,
  Shield,
  Target
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface CallStatistics {
  most_incoming_calls: Array<{ contact: string; count: number }>;
  most_outgoing_calls: Array<{ contact: string; count: number }>;
  most_contacted_numbers: Array<{ contact: string; count: number }>;
  call_duration_stats: {
    average_duration: number;
    total_duration: number;
  };
  missed_calls: number;
  answered_calls: number;
  total_call_duration: number;
}

interface CommunicationPatterns {
  daily_activity: Record<string, number>;
  hourly_patterns: Record<string, number>;
  contact_frequency: Record<string, number>;
  suspicious_patterns: Array<{
    type: string;
    contact: string;
    frequency: number;
    description: string;
  }>;
}

interface ForensicInsights {
  location_patterns: Array<any>;
  device_information: Array<any>;
  timeline_gaps: Array<any>;
  anomalies: Array<any>;
}

interface CaseSummary {
  total_cases: number;
  total_evidence_files: number;
  data_types: Record<string, number>;
  date_range: Record<string, any>;
}

interface DetailedAnalytics {
  call_statistics: CallStatistics;
  communication_patterns: CommunicationPatterns;
  forensic_insights: ForensicInsights;
  case_summary: CaseSummary;
}

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
  const [detailedAnalytics, setDetailedAnalytics] = useState<DetailedAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailedLoading, setDetailedLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    loadAnalytics();
    loadDetailedAnalytics();
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

  const loadDetailedAnalytics = async () => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    
    setDetailedLoading(true);
    try {
      const url = selectedCaseId 
        ? `http://localhost:8000/database/analytics/detailed?case_id=${selectedCaseId}`
        : `http://localhost:8000/database/analytics/detailed`;
        
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Failed to load detailed analytics');
      }
      
      const data = await response.json();
      if (data.success) {
        setDetailedAnalytics(data.analytics);
      }
    } catch (error) {
      console.error('Error loading detailed analytics:', error);
      toast({
        title: "Error loading detailed analytics",
        description: "Some analytics features may not be available.",
        variant: "destructive",
      });
    } finally {
      setDetailedLoading(false);
    }
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
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

      {/* Tabs for Different Analytics Views */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="calls">Call Analytics</TabsTrigger>
          <TabsTrigger value="patterns">Patterns</TabsTrigger>
          <TabsTrigger value="forensics">Forensic Insights</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">

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
        </TabsContent>

        {/* Call Analytics Tab */}
        <TabsContent value="calls" className="space-y-6">
          {detailedLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Loading call analytics...</span>
            </div>
          ) : detailedAnalytics ? (
            <>
              {/* Call Statistics Overview */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Answered Calls</p>
                        <p className="text-2xl font-bold text-green-600">{formatNumber(detailedAnalytics.call_statistics.answered_calls)}</p>
                      </div>
                      <PhoneCall className="h-8 w-8 text-green-600" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Missed Calls</p>
                        <p className="text-2xl font-bold text-red-600">{formatNumber(detailedAnalytics.call_statistics.missed_calls)}</p>
                      </div>
                      <Phone className="h-8 w-8 text-red-600" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Avg Duration</p>
                        <p className="text-2xl font-bold text-blue-600">{formatDuration(Math.round(detailedAnalytics.call_statistics.call_duration_stats.average_duration))}</p>
                      </div>
                      <Timer className="h-8 w-8 text-blue-600" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Total Duration</p>
                        <p className="text-2xl font-bold text-purple-600">{formatDuration(detailedAnalytics.call_statistics.total_call_duration)}</p>
                      </div>
                      <Clock className="h-8 w-8 text-purple-600" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Call Direction Analysis */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <PhoneIncoming className="h-5 w-5 text-green-600" />
                      Most Incoming Calls
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {detailedAnalytics.call_statistics.most_incoming_calls.slice(0, 5).map((call, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                              <PhoneIncoming className="h-4 w-4 text-green-600" />
                            </div>
                            <span className="font-medium">{call.contact}</span>
                          </div>
                          <Badge variant="secondary">{call.count} calls</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <PhoneOutgoing className="h-5 w-5 text-blue-600" />
                      Most Outgoing Calls
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {detailedAnalytics.call_statistics.most_outgoing_calls.slice(0, 5).map((call, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                              <PhoneOutgoing className="h-4 w-4 text-blue-600" />
                            </div>
                            <span className="font-medium">{call.contact}</span>
                          </div>
                          <Badge variant="secondary">{call.count} calls</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="h-5 w-5 text-purple-600" />
                      Most Contacted
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {detailedAnalytics.call_statistics.most_contacted_numbers.slice(0, 5).map((contact, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                              <Target className="h-4 w-4 text-purple-600" />
                            </div>
                            <span className="font-medium">{contact.contact}</span>
                          </div>
                          <Badge variant="secondary">{contact.count} total</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <div className="text-center py-12">
              <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No call data available</p>
            </div>
          )}
        </TabsContent>

        {/* Communication Patterns Tab */}
        <TabsContent value="patterns" className="space-y-6">
          {detailedLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Analyzing patterns...</span>
            </div>
          ) : detailedAnalytics && detailedAnalytics.communication_patterns ? (
            <>
              {/* Suspicious Patterns */}
              {detailedAnalytics.communication_patterns.suspicious_patterns && detailedAnalytics.communication_patterns.suspicious_patterns.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5 text-red-600" />
                      Suspicious Patterns
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {detailedAnalytics.communication_patterns.suspicious_patterns.map((pattern, index) => (
                        <div key={index} className="p-4 border-l-4 border-red-500 bg-red-50 rounded-lg">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium text-red-800">{pattern.description}</p>
                              <p className="text-sm text-red-600">Contact: {pattern.contact}</p>
                            </div>
                            <Badge variant="destructive">{pattern.frequency} occurrences</Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Activity Patterns */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Calendar className="h-5 w-5" />
                      Daily Activity
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {detailedAnalytics.communication_patterns.daily_activity && Object.entries(detailedAnalytics.communication_patterns.daily_activity)
                        .sort(([a], [b]) => b.localeCompare(a))
                        .slice(0, 7)
                        .map(([date, count], index) => (
                          <div key={index} className="flex items-center justify-between">
                            <span className="font-medium">{date}</span>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">{count} activities</Badge>
                              <Progress value={(count / Math.max(...Object.values(detailedAnalytics.communication_patterns.daily_activity))) * 100} className="w-16" />
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      Hourly Patterns
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {detailedAnalytics.communication_patterns.hourly_patterns && Object.entries(detailedAnalytics.communication_patterns.hourly_patterns)
                        .sort(([a], [b]) => a.localeCompare(b))
                        .slice(0, 8)
                        .map(([hour, count], index) => (
                          <div key={index} className="flex items-center justify-between">
                            <span className="font-medium">{hour}:00</span>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">{count} activities</Badge>
                              <Progress value={(count / Math.max(...Object.values(detailedAnalytics.communication_patterns.hourly_patterns))) * 100} className="w-16" />
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <div className="text-center py-12">
              <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No pattern data available</p>
            </div>
          )}
        </TabsContent>

        {/* Forensic Insights Tab */}
        <TabsContent value="forensics" className="space-y-6">
          {detailedLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Generating forensic insights...</span>
            </div>
          ) : detailedAnalytics && detailedAnalytics.case_summary ? (
            <>
              {/* Case Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Case Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600 mb-2">
                        {detailedAnalytics.case_summary.total_cases}
                      </div>
                      <p className="text-sm text-muted-foreground">Total Cases</p>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-green-600 mb-2">
                        {detailedAnalytics.case_summary.total_evidence_files}
                      </div>
                      <p className="text-sm text-muted-foreground">Evidence Files</p>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-purple-600 mb-2">
                        {detailedAnalytics.case_summary.data_types ? Object.keys(detailedAnalytics.case_summary.data_types).length : 0}
                      </div>
                      <p className="text-sm text-muted-foreground">Data Types</p>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-orange-600 mb-2">
                        {(detailedAnalytics.call_statistics?.answered_calls || 0) + (detailedAnalytics.call_statistics?.missed_calls || 0)}
                      </div>
                      <p className="text-sm text-muted-foreground">Total Calls</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Data Types Distribution */}
              {detailedAnalytics.case_summary.data_types && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Database className="h-5 w-5" />
                      Data Types Distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(detailedAnalytics.case_summary.data_types).map(([type, count], index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center gap-3">
                            {getDataSourceIcon(type)}
                            <span className="font-medium capitalize">{type}</span>
                          </div>
                          <Badge variant="secondary">{count} files</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No forensic data available</p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Analytics;