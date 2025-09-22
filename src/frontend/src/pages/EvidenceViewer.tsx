import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { 
  Search, 
  Filter, 
  Calendar, 
  Eye, 
  Download, 
  Share2, 
  FileText, 
  Image, 
  FileAudio, 
  Video,
  Smartphone,
  Mail,
  Clock,
  MapPin,
  Users,
  Zap,
  Brain,
  TrendingUp,
  AlertCircle,
  RefreshCw,
  Upload
} from 'lucide-react';

// Import API services
import { evidenceService, EnhancedEvidence, TimelineEvent } from '@/services/evidenceService';
import { caseService } from '@/services/caseService';
import { ApiError } from '@/services/api';

// Use the EnhancedEvidence type from the service
type Evidence = EnhancedEvidence;

interface Timeline {
  timestamp: string;
  event: string;
  evidence: string[];
  importance: 'low' | 'medium' | 'high' | 'critical';
}

const EvidenceViewer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [evidenceList, setEvidenceList] = useState<Evidence[]>([]);
  const [timeline, setTimeline] = useState<Timeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'grid' | 'timeline' | 'relationships'>('grid');
  const [selectedCaseId, setSelectedCaseId] = useState<string>('');
  const [refreshing, setRefreshing] = useState(false);

  // Get case ID from URL params or use a default
  useEffect(() => {
    const resolveAndSetCaseId = async () => {
      try {
        // In a real app, you'd get this from router params
        // For now, we'll use a default case number and resolve it to ID
        const caseNumberOrId = new URLSearchParams(window.location.search).get('caseId') || 'DEMO-2024-001';
        
        // Resolve case number to case ID if needed
        const resolvedCaseId = await caseService.resolveCaseId(caseNumberOrId);
        setSelectedCaseId(resolvedCaseId);
      } catch (error) {
        console.error('Failed to resolve case ID:', error);
        setError(`Failed to find case: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    };

    resolveAndSetCaseId();
  }, []);

  // Fetch evidence data from API
  useEffect(() => {
    if (!selectedCaseId) return;
    
    const fetchEvidenceData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch evidence and timeline data
        const [evidenceData, timelineData] = await Promise.allSettled([
          evidenceService.getEnhancedCaseEvidence(selectedCaseId),
          evidenceService.getEvidenceTimeline(selectedCaseId)
        ]);

        // Handle evidence data
        if (evidenceData.status === 'fulfilled') {
          setEvidenceList(evidenceData.value);
        } else {
          console.error('Failed to fetch evidence:', evidenceData.reason);
          throw new Error('Failed to load evidence data');
        }

        // Handle timeline data
        if (timelineData.status === 'fulfilled') {
          const timelineEvents = timelineData.value.timeline_events.map((event: TimelineEvent) => ({
            timestamp: event.timestamp,
            event: event.event,
            evidence: event.evidence,
            importance: event.importance
          }));
          setTimeline(timelineEvents);
        } else {
          console.warn('Timeline data not available, using empty timeline');
          setTimeline([]);
        }

      } catch (err) {
        console.error('Error fetching evidence data:', err);
        let errorMessage = 'Failed to load evidence data';
        
        if (err instanceof ApiError) {
          errorMessage = `API Error: ${err.message}`;
        } else if (err instanceof Error) {
          errorMessage = err.message;
        }
        
        setError(errorMessage);
        setEvidenceList([]);
        setTimeline([]);
      } finally {
        setLoading(false);
      }
    };

    fetchEvidenceData();
  }, [selectedCaseId]);

  // Refresh data
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      const evidenceData = await evidenceService.getEnhancedCaseEvidence(selectedCaseId);
      setEvidenceList(evidenceData);
      setError(null);
    } catch (err) {
      console.error('Error refreshing evidence:', err);
      setError('Failed to refresh evidence data');
    } finally {
      setRefreshing(false);
    }
  };

  const getEvidenceIcon = (type: Evidence['type']) => {
    switch (type) {
      case 'document': return <FileText className="h-4 w-4" />;
      case 'image': return <Image className="h-4 w-4" />;
      case 'audio': return <FileAudio className="h-4 w-4" />;
      case 'video': return <Video className="h-4 w-4" />;
      case 'call_log': return <Smartphone className="h-4 w-4" />;
      case 'email': return <Mail className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getImportanceColor = (importance: Timeline['importance']) => {
    switch (importance) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const filteredEvidence = evidenceList.filter(evidence =>
    evidence.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    evidence.summary.toLowerCase().includes(searchTerm.toLowerCase()) ||
    evidence.entities.some(entity => entity.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
        <p className="text-muted-foreground">Loading evidence data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <AlertCircle className="h-16 w-16 text-destructive" />
        <div className="text-center">
          <h3 className="text-lg font-semibold text-destructive">Error Loading Evidence</h3>
          <p className="text-muted-foreground mt-2">{error}</p>
          <Button 
            onClick={handleRefresh} 
            className="mt-4"
            disabled={refreshing}
          >
            {refreshing ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Evidence Viewer</h1>
          <p className="text-muted-foreground">
            AI-powered forensic evidence analysis and visualization - Case: {selectedCaseId}
          </p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleRefresh}
            disabled={refreshing}
          >
            {refreshing ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
          <Button variant="outline" size="sm">
            <Share2 className="h-4 w-4 mr-2" />
            Share Analysis
          </Button>
        </div>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4 items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search evidence, entities, or AI insights..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button variant="outline" size="sm">
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
            <Button variant="outline" size="sm">
              <Calendar className="h-4 w-4 mr-2" />
              Timeline
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* View Selector */}
      <Tabs value={selectedView} onValueChange={(value) => setSelectedView(value as any)}>
        <TabsList>
          <TabsTrigger value="grid">Grid View</TabsTrigger>
          <TabsTrigger value="timeline">Timeline View</TabsTrigger>
          <TabsTrigger value="relationships">Relationship Map</TabsTrigger>
        </TabsList>

        <TabsContent value="grid" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Evidence List */}
            <div className="lg:col-span-2 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Evidence Collection ({filteredEvidence.length})
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    {filteredEvidence.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full space-y-4 text-center py-8">
                        <FileText className="h-16 w-16 text-muted-foreground/50" />
                        <div>
                          <h3 className="text-lg font-medium text-muted-foreground">No Evidence Found</h3>
                          <p className="text-sm text-muted-foreground mt-1">
                            {searchTerm 
                              ? `No evidence matches "${searchTerm}"`
                              : `No evidence available for case ${selectedCaseId}`
                            }
                          </p>
                          {!searchTerm && (
                            <Button 
                              variant="outline" 
                              size="sm" 
                              className="mt-4"
                              onClick={() => {/* TODO: Implement upload */}}
                            >
                              <Upload className="h-4 w-4 mr-2" />
                              Upload Evidence
                            </Button>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                      {filteredEvidence.map((evidence) => (
                        <Card 
                          key={evidence.id}
                          className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                            selectedEvidence?.id === evidence.id ? 'ring-2 ring-primary' : ''
                          }`}
                          onClick={() => setSelectedEvidence(evidence)}
                        >
                          <CardContent className="p-4">
                            <div className="flex items-start gap-3">
                              <div className="p-2 bg-muted rounded">
                                {getEvidenceIcon(evidence.type)}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <h4 className="font-medium truncate">{evidence.name}</h4>
                                  <Badge variant="secondary" className="text-xs">
                                    AI: {Math.round(evidence.aiConfidence * 100)}%
                                  </Badge>
                                </div>
                                <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                                  {evidence.summary}
                                </p>
                                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                  <span className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {new Date(evidence.timestamp).toLocaleDateString()}
                                  </span>
                                  <span className="flex items-center gap-1">
                                    <Users className="h-3 w-3" />
                                    {evidence.entities.length} entities
                                  </span>
                                  <span className="flex items-center gap-1">
                                    <TrendingUp className="h-3 w-3" />
                                    {evidence.relationships} connections
                                  </span>
                                </div>
                                <div className="flex gap-1 mt-2">
                                  {evidence.tags.slice(0, 3).map((tag) => (
                                    <Badge key={tag} variant="outline" className="text-xs">
                                      {tag}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>

            {/* Evidence Details */}
            <div className="space-y-4">
              {selectedEvidence ? (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Eye className="h-5 w-5" />
                        Evidence Details
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <h4 className="font-medium mb-2">{selectedEvidence.name}</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="text-muted-foreground">Type:</div>
                          <div className="capitalize">{selectedEvidence.type}</div>
                          <div className="text-muted-foreground">Size:</div>
                          <div>{selectedEvidence.size}</div>
                          <div className="text-muted-foreground">Source:</div>
                          <div>{selectedEvidence.source}</div>
                          <div className="text-muted-foreground">AI Confidence:</div>
                          <div className="flex items-center gap-2">
                            {Math.round(selectedEvidence.aiConfidence * 100)}%
                            <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-primary transition-all"
                                style={{ width: `${selectedEvidence.aiConfidence * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>

                      <Separator />

                      <div>
                        <h5 className="font-medium mb-2 flex items-center gap-2">
                          <Brain className="h-4 w-4" />
                          AI Analysis
                        </h5>
                        <p className="text-sm text-muted-foreground">
                          {selectedEvidence.summary}
                        </p>
                      </div>

                      <Separator />

                      <div>
                        <h5 className="font-medium mb-2 flex items-center gap-2">
                          <Users className="h-4 w-4" />
                          Extracted Entities
                        </h5>
                        <div className="flex flex-wrap gap-1">
                          {selectedEvidence.entities.map((entity, index) => (
                            <Badge key={index} variant="secondary" className="text-xs">
                              {entity}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      <Separator />

                      <div>
                        <h5 className="font-medium mb-2">Metadata</h5>
                        <div className="space-y-1 text-sm">
                          {Object.entries(selectedEvidence.metadata).map(([key, value]) => (
                            <div key={key} className="grid grid-cols-2 gap-2">
                              <div className="text-muted-foreground capitalize">
                                {key.replace(/([A-Z])/g, ' $1').trim()}:
                              </div>
                              <div>{Array.isArray(value) ? value.join(', ') : String(value)}</div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="flex gap-2 pt-4">
                        <Button size="sm" className="flex-1">
                          <Eye className="h-4 w-4 mr-2" />
                          Preview
                        </Button>
                        <Button variant="outline" size="sm">
                          <Download className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </>
              ) : (
                <Card>
                  <CardContent className="text-center py-12">
                    <Eye className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="font-medium mb-2">Select Evidence</h3>
                    <p className="text-sm text-muted-foreground">
                      Choose an evidence item to view detailed analysis and AI insights
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="timeline" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Evidence Timeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {timeline.map((event, index) => (
                  <div key={index} className="flex gap-4">
                    <div className="flex flex-col items-center">
                      <div className={`w-3 h-3 rounded-full ${getImportanceColor(event.importance)}`} />
                      {index < timeline.length - 1 && <div className="w-px h-16 bg-border mt-2" />}
                    </div>
                    <div className="flex-1 pb-8">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium">{event.event}</h4>
                        <Badge 
                          variant={event.importance === 'critical' ? 'destructive' : 'secondary'}
                          className="text-xs"
                        >
                          {event.importance}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        {new Date(event.timestamp).toLocaleString()}
                      </p>
                      <div className="flex gap-1">
                        {event.evidence.map((evidenceId) => {
                          const evidence = evidenceList.find(e => e.id === evidenceId);
                          return evidence ? (
                            <Badge key={evidenceId} variant="outline" className="text-xs">
                              {evidence.name}
                            </Badge>
                          ) : null;
                        })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="relationships" className="space-y-4">
          <Card>
            <CardContent className="text-center py-12">
              <Zap className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="font-medium mb-2">Relationship Visualization</h3>
              <p className="text-sm text-muted-foreground">
                Interactive network graph showing connections between evidence, entities, and events
              </p>
              <Button className="mt-4">
                <Brain className="h-4 w-4 mr-2" />
                Generate AI Network Map
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EvidenceViewer;