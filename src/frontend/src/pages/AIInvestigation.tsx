import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { 
  Brain, 
  Zap, 
  Target, 
  Clock, 
  TrendingUp, 
  Network,
  Search,
  Eye,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Bot,
  Sparkles,
  GitBranch,
  Activity,
  Users,
  MapPin,
  Calendar,
  FileText,
  MessageSquare,
  Lightbulb,
  Shield,
  Star
} from 'lucide-react';
import { caseService } from '@/services/caseService';
import { useNavigate } from 'react-router-dom';

interface TimelineEvent {
  id: string;
  timestamp: string;
  event: string;
  description: string;
  importance: 'low' | 'medium' | 'high' | 'critical';
  entities: string[];
  evidence_ids: string[];
}

interface Pattern {
  id: string;
  type: string;
  description: string;
  confidence: number;
  entities: string[];
  timeline_events: string[];
}

interface SupremeAnalysis {
  summary: string;
  key_findings: string[];
  investigative_leads: string[];
  recommendations: string[];
  evidence_gaps: string[];
  next_steps: string[];
  risk_assessment: string;
  confidence_score: number;
  patterns_detected?: Pattern[]; // Add patterns detected from step 2
}

interface AIAnalysisResult {
  timeline: TimelineEvent[];
  patterns: Pattern[];
  supreme_analysis: SupremeAnalysis;
  case_summary: string;
  processing_stats: {
    entities_analyzed: number;
    evidence_processed: number;
    patterns_found: number;
    timeline_events: number;
  };
}

const AIInvestigation: React.FC = () => {
  const navigate = useNavigate();
  const [selectedCaseId, setSelectedCaseId] = useState<string>('');
  const [caseInfo, setCaseInfo] = useState<{ case_number: string; title: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AIAnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Get case ID from localStorage
  useEffect(() => {
    console.log('🔍 AI Investigation - Initializing...');
    
    const caseId = localStorage.getItem('selectedCaseId');
    const caseNumber = localStorage.getItem('selectedCaseNumber');
    
    console.log('localStorage check:', {
      selectedCaseId: caseId,
      selectedCaseNumber: caseNumber
    });
    
    if (caseId) {
      console.log('✅ Case ID found, setting up AI Investigation for case:', caseId);
      setSelectedCaseId(caseId);
      if (caseNumber) {
        setCaseInfo({ case_number: caseNumber, title: 'Loading...' });
      }
    } else {
      console.log('❌ No case selected for AI Investigation');
      setError('No case selected. Please select a case from the dashboard first.');
    }
  }, []);

  const runFullAIAnalysis = async () => {
    if (!selectedCaseId) {
      console.log('❌ No case ID selected for analysis');
      return;
    }
    
    console.log('🚀 Starting Full AI Analysis for case:', selectedCaseId);
    setLoading(true);
    setError(null);
    setAnalysisProgress(0);

    try {
      // Step 1: Start timeline generation
      setAnalysisProgress(20);
      console.log('� Step 1: Starting AI timeline generation...');
      
      const timelineUrl = `http://localhost:8000/cases/${selectedCaseId}/evidence/analyze-patterns`;
      console.log('🔗 Timeline API URL:', timelineUrl);
      
      const timelineResponse = await fetch(timelineUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_type: 'timeline_generation' })
      });
      
      console.log('📈 Timeline response status:', timelineResponse.status);
      
      if (!timelineResponse.ok) {
        const errorText = await timelineResponse.text();
        console.error('❌ Timeline generation failed:', errorText);
        throw new Error(`Timeline generation failed: ${timelineResponse.status} ${errorText}`);
      }
      
      const timelineData = await timelineResponse.json();
      console.log('✅ Timeline data received:', timelineData);
      setAnalysisProgress(40);

      // Step 2: Pattern detection
      console.log('🔍 Step 2: Starting pattern detection...');
      
      const patternResponse = await fetch(timelineUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_type: 'pattern_detection' })
      });
      
      console.log('📊 Pattern response status:', patternResponse.status);
      
      if (!patternResponse.ok) {
        const errorText = await patternResponse.text();
        console.error('❌ Pattern detection failed:', errorText);
        throw new Error(`Pattern detection failed: ${patternResponse.status} ${errorText}`);
      }
      
      const patternData = await patternResponse.json();
      console.log('✅ Pattern data received:', patternData);
      setAnalysisProgress(60);

      // Step 3: Supreme Forensic Agent analysis
      console.log('🧠 Step 3: Starting Supreme Forensic Agent analysis...');
      
      const supremeUrl = `http://localhost:8000/cases/${selectedCaseId}/evidence/supreme-analysis`;
      console.log('🔗 Supreme API URL:', supremeUrl);
      
      const supremeResponse = await fetch(supremeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: 'Provide comprehensive forensic analysis of this case including timeline, patterns, and investigative recommendations',
          analysis_mode: 'overview'
        })
      });
      
      console.log('🧠 Supreme response status:', supremeResponse.status);
      
      if (!supremeResponse.ok) {
        const errorText = await supremeResponse.text();
        console.error('❌ Supreme analysis failed:', errorText);
        throw new Error(`Supreme analysis failed: ${supremeResponse.status} ${errorText}`);
      }
      
      const supremeData = await supremeResponse.json();
      console.log('✅ Supreme data received:', supremeData);
      setAnalysisProgress(80);

      // Step 4: Network analysis
      console.log('🌐 Step 4: Starting network analysis...');
      
      const networkResponse = await fetch(timelineUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_type: 'network_analysis' })
      });
      
      console.log('🌐 Network response status:', networkResponse.status);
      const networkData = networkResponse.ok ? await networkResponse.json() : null;
      console.log('✅ Network data received:', networkData);
      setAnalysisProgress(100);

      // Combine all results
      console.log('🔄 Step 5: Combining results...');
      
      // Transform pattern data to match frontend interface
      const transformPatterns = (patterns: any[]): Pattern[] => {
        return patterns.map((pattern, index) => ({
          id: pattern.id || `pattern-${index}`,
          type: pattern.pattern_type || pattern.type || 'Unknown Pattern',
          description: pattern.description || 'No description available',
          confidence: Math.round((pattern.confidence || 0) * 100), // Convert to percentage
          entities: pattern.entities || [],
          timeline_events: pattern.timeline_events || []
        }));
      };

      const transformedPatterns = transformPatterns(
        patternData.patterns_detected || patternData.patterns_found || []
      );
      
      const combinedResults: AIAnalysisResult = {
        timeline: timelineData.results?.timeline || [],
        patterns: transformedPatterns,
        supreme_analysis: {
          summary: supremeData.structured_analysis?.summary || 'Analysis completed',
          key_findings: supremeData.structured_analysis?.key_findings || [],
          investigative_leads: supremeData.investigation_recommendations || [],
          recommendations: supremeData.investigation_recommendations || [],
          evidence_gaps: supremeData.evidence_gaps || [],
          next_steps: supremeData.next_steps || [],
          risk_assessment: supremeData.structured_analysis?.risk_assessment || 'Medium',
          confidence_score: supremeData.confidence_score || 85,
          // Include transformed patterns from step 2 in supreme analysis
          patterns_detected: transformedPatterns
        },
        case_summary: supremeData.structured_analysis?.case_summary || 'Comprehensive AI analysis completed',
        processing_stats: {
          entities_analyzed: timelineData.workflow_steps?.length || 0,
          evidence_processed: patternData.workflow_steps?.length || 0,
          patterns_found: transformedPatterns.length,
          timeline_events: timelineData.results?.timeline?.length || 0
        }
      };

      console.log('🎉 Final combined results:', combinedResults);
      setAnalysisResults(combinedResults);
      console.log('✅ Analysis completed successfully!');

    } catch (error) {
      console.error('💥 AI analysis failed with error:', error);
      setError(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600';
    if (confidence >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (error && !selectedCaseId) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center min-h-[50vh]">
          <Card className="w-full max-w-md">
            <CardContent className="text-center py-8">
              <AlertTriangle className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
              <h3 className="font-medium mb-2">No Case Selected</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Please select a case to run AI investigation analysis.
              </p>
              <div className="space-y-2">
                <Button onClick={() => navigate('/')} className="w-full">
                  Select a Case
                </Button>
                <Button 
                  variant="outline" 
                  onClick={() => navigate('/dashboard')} 
                  className="w-full"
                >
                  Back to Dashboard
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Brain className="h-8 w-8 text-purple-600" />
            AI Investigation Center
          </h1>
          <p className="text-muted-foreground mt-1">
            Advanced AI-powered forensic analysis and investigation
          </p>
          {caseInfo && (
            <div className="mt-2 flex items-center gap-2">
              <Badge variant="outline" className="text-sm">
                📂 Case {caseInfo.case_number}: {caseInfo.title}
              </Badge>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => navigate('/')}
                className="text-xs"
              >
                Change Case
              </Button>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          <Button 
            onClick={runFullAIAnalysis}
            disabled={loading || !selectedCaseId}
            size="lg"
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
          >
            {loading ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="h-5 w-5 mr-2" />
                Run Full AI Analysis
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Analysis Progress */}
      {loading && (
        <Card>
          <CardContent className="py-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">AI Analysis Progress</h3>
                <span className="text-sm text-muted-foreground">{analysisProgress}%</span>
              </div>
              <Progress value={analysisProgress} className="h-2" />
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Bot className="h-4 w-4 animate-pulse" />
                {analysisProgress < 20 && "Initializing AI engines..."}
                {analysisProgress >= 20 && analysisProgress < 40 && "Generating timeline..."}
                {analysisProgress >= 40 && analysisProgress < 60 && "Detecting patterns..."}
                {analysisProgress >= 60 && analysisProgress < 80 && "Running Supreme Forensic Agent..."}
                {analysisProgress >= 80 && analysisProgress < 100 && "Analyzing networks..."}
                {analysisProgress === 100 && "Analysis complete!"}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {error && selectedCaseId && (
        <Card className="border-red-200">
          <CardContent className="py-4">
            <div className="flex items-center gap-2 text-red-600">
              <AlertTriangle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Analysis Results */}
      {analysisResults && (
        <>
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Timeline Events</p>
                    <p className="text-2xl font-bold">{analysisResults.processing_stats.timeline_events}</p>
                  </div>
                  <Clock className="h-8 w-8 text-blue-500" />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Patterns Found</p>
                    <p className="text-2xl font-bold">{analysisResults.processing_stats.patterns_found}</p>
                  </div>
                  <Target className="h-8 w-8 text-green-500" />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Evidence Processed</p>
                    <p className="text-2xl font-bold">{analysisResults.processing_stats.evidence_processed}</p>
                  </div>
                  <FileText className="h-8 w-8 text-orange-500" />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Confidence Score</p>
                    <p className={`text-2xl font-bold ${getConfidenceColor(analysisResults.supreme_analysis.confidence_score)}`}>
                      {analysisResults.supreme_analysis.confidence_score}%
                    </p>
                  </div>
                  <Star className="h-8 w-8 text-purple-500" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Analysis Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="timeline">Timeline</TabsTrigger>
              <TabsTrigger value="patterns">Patterns</TabsTrigger>
              <TabsTrigger value="insights">AI Insights</TabsTrigger>
              <TabsTrigger value="recommendations">Actions</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <MessageSquare className="h-5 w-5" />
                      Case Summary
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed">{analysisResults.case_summary}</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Risk Assessment
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span>Risk Level:</span>
                        <Badge variant={analysisResults.supreme_analysis.risk_assessment === 'High' ? 'destructive' : 'secondary'}>
                          {analysisResults.supreme_analysis.risk_assessment}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Confidence:</span>
                        <span className={`font-bold ${getConfidenceColor(analysisResults.supreme_analysis.confidence_score)}`}>
                          {analysisResults.supreme_analysis.confidence_score}%
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5" />
                    Key Findings
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {analysisResults.supreme_analysis.key_findings}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Timeline Tab */}
            <TabsContent value="timeline" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <GitBranch className="h-5 w-5" />
                    AI-Generated Timeline
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    <div className="space-y-4">
                      {analysisResults.timeline.map((event, index) => (
                        <div key={event.id} className="flex gap-4">
                          <div className="flex flex-col items-center">
                            <div className={`w-3 h-3 rounded-full ${getImportanceColor(event.importance)}`} />
                            {index < analysisResults.timeline.length - 1 && (
                              <div className="w-px h-16 bg-gray-200 mt-2" />
                            )}
                          </div>
                          <div className="flex-1 pb-4">
                            <div className="flex items-center justify-between mb-1">
                              <h4 className="font-medium">{event.event}</h4>
                              <Badge variant="outline" className="text-xs">
                                {event.importance}
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mb-2">
                              {new Date(event.timestamp).toLocaleString()}
                            </p>
                            <p className="text-sm">{event.description}</p>
                            {event.entities.length > 0 && (
                              <div className="flex gap-1 mt-2">
                                {event.entities.map((entity, idx) => (
                                  <Badge key={idx} variant="secondary" className="text-xs">
                                    {entity}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Patterns Tab */}
            <TabsContent value="patterns" className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {analysisResults.patterns.map((pattern) => (
                  <Card key={pattern.id}>
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <span className="flex items-center gap-2">
                          <Network className="h-5 w-5" />
                          {pattern.type}
                        </span>
                        <Badge variant="outline" className={getConfidenceColor(pattern.confidence)}>
                          {pattern.confidence}% confidence
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm mb-3">{pattern.description}</p>
                      {pattern.entities.length > 0 && (
                        <div className="space-y-2">
                          <p className="text-xs font-medium text-muted-foreground">Related Entities:</p>
                          <div className="flex flex-wrap gap-1">
                            {pattern.entities.map((entity, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs">
                                {entity}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* AI Insights Tab */}
            <TabsContent value="insights" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    Supreme Forensic Agent Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2">Summary</h4>
                      <p className="text-sm text-muted-foreground">
                        {analysisResults.supreme_analysis.summary}
                      </p>
                    </div>
                    
                    <Separator />
                    
                    <div>
                      <h4 className="font-medium mb-2">Investigative Leads</h4>
                      <div className="space-y-2">
                        {analysisResults.supreme_analysis.investigative_leads.map((lead, index) => (
                          <div key={index} className="flex items-start gap-2">
                            <Search className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm">{lead}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Add Patterns Detected Section */}
                    {analysisResults.supreme_analysis.patterns_detected && analysisResults.supreme_analysis.patterns_detected.length > 0 && (
                      <>
                        <Separator />
                        <div>
                          <h4 className="font-medium mb-2 flex items-center gap-2">
                            <Network className="h-4 w-4 text-green-500" />
                            Patterns Detected
                          </h4>
                          <div className="space-y-2">
                            {analysisResults.supreme_analysis.patterns_detected.map((pattern, index) => (
                              <div key={index} className="flex items-start gap-2 p-2 bg-green-50 rounded-lg">
                                <TrendingUp className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                                <div className="flex-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-green-900">{pattern.type}</span>
                                    <Badge variant="outline" className="text-xs text-green-700 border-green-300">
                                      {pattern.confidence}% confidence
                                    </Badge>
                                  </div>
                                  <p className="text-sm text-green-700 mt-1">{pattern.description}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Recommendations Tab */}
            <TabsContent value="recommendations" className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Recommended Actions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {analysisResults.supreme_analysis.recommendations.map((recommendation, index) => (
                        <div key={index} className="flex items-start gap-3 p-3 bg-blue-50 rounded-lg">
                          <Activity className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                          <div>
                            <p className="text-sm font-medium text-blue-900">Action {index + 1}</p>
                            <p className="text-sm text-blue-700">{recommendation}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Next Steps
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {analysisResults.supreme_analysis.next_steps.map((step, index) => (
                        <div key={index} className="flex items-start gap-3 p-3 bg-green-50 rounded-lg">
                          <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
                          <div>
                            <p className="text-sm font-medium text-green-900">Step {index + 1}</p>
                            <p className="text-sm text-green-700">{step}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5" />
                    Evidence Gaps
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {analysisResults.supreme_analysis.evidence_gaps.map((gap, index) => (
                      <div key={index} className="flex items-start gap-3 p-3 bg-orange-50 rounded-lg">
                        <AlertTriangle className="h-5 w-5 text-orange-600 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium text-orange-900">Gap {index + 1}</p>
                          <p className="text-sm text-orange-700">{gap}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}

      {/* Initial State */}
      {!analysisResults && !loading && !error && (
        <Card className="text-center py-12">
          <CardContent>
            <Bot className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">Ready for AI Analysis</h3>
            <p className="text-muted-foreground mb-6">
              Click "Run Full AI Analysis" to start comprehensive forensic investigation using our advanced AI engines.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
              <div className="p-4 bg-purple-50 rounded-lg">
                <Brain className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                <h4 className="font-medium text-purple-900">GPT-5 Intelligence Engine</h4>
                <p className="text-sm text-purple-700">Timeline generation & pattern detection</p>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <Shield className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                <h4 className="font-medium text-blue-900">Supreme Forensic Agent</h4>
                <p className="text-sm text-blue-700">Expert investigation & recommendations</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AIInvestigation;