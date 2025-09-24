import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { 
  Network, 
  Users, 
  Phone, 
  Mail, 
  Clock, 
  TrendingUp, 
  Brain, 
  Zap,
  Search,
  Filter,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Download,
  Share2,
  Target,
  GitBranch,
  Activity
} from 'lucide-react';

interface NetworkNode {
  id: string;
  label: string;
  type: 'person' | 'phone' | 'email' | 'location' | 'device' | 'account';
  importance: number;
  connections: number;
  metadata: Record<string, any>;
  x?: number;
  y?: number;
}

interface NetworkEdge {
  id: string;
  source: string;
  target: string;
  type: 'communication' | 'ownership' | 'location' | 'transaction' | 'association';
  weight: number;
  frequency: number;
  lastActivity: string;
  metadata: Record<string, any>;
}

interface CommunicationPattern {
  participants: string[];
  frequency: number;
  pattern: 'regular' | 'burst' | 'suspicious' | 'normal';
  timeRange: string;
  suspicionScore: number;
}

const NetworkAnalysis: React.FC = () => {
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [edges, setEdges] = useState<NetworkEdge[]>([]);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [timeRange, setTimeRange] = useState([0, 100]);
  const [analysisMode, setAnalysisMode] = useState<'static' | 'temporal' | 'simulation'>('static');
  const [isPlaying, setIsPlaying] = useState(false);
  const [patterns, setPatterns] = useState<CommunicationPattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentCase, setCurrentCase] = useState<string>('');
  const [error, setError] = useState<string>('');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Get case ID from URL params or use a default case
  const getCaseId = () => {
    const params = new URLSearchParams(window.location.search);
    return params.get('case') || 'ENHANCED-2024-001'; // Default case ID
  };

  // Fetch network data from backend API
  useEffect(() => {
    const fetchNetworkData = async () => {
      try {
        setLoading(true);
        const caseId = getCaseId();
        
        // First, get available cases
        const casesResponse = await fetch('http://localhost:8000/cases');
        const casesData = await casesResponse.json();
        
        // Use the first available case if default doesn't exist
        const actualCaseId = casesData.length > 0 ? casesData[0].id : caseId;
        setCurrentCase(actualCaseId);
        
        // Fetch network analysis data
        const response = await fetch(`http://localhost:8000/cases/${actualCaseId}/network/data`);
        
        if (!response.ok) {
          throw new Error(`Network API returned ${response.status}`);
        }
        
        const networkData = await response.json();
        console.log('Network data received:', networkData);
        
        // Transform backend data to frontend format
        const transformedNodes: NetworkNode[] = (networkData.entities || []).map((entity: any, index: number) => ({
          id: entity.id || `entity_${index}`,
          label: entity.name || entity.label || `Entity ${index}`,
          type: mapEntityType(entity.type),
          importance: entity.importance || 0.5,
          connections: entity.connections || 1,
          metadata: entity.properties || entity.metadata || {},
          x: Math.random() * 800 + 100,
          y: Math.random() * 600 + 100
        }));

        const transformedEdges: NetworkEdge[] = (networkData.network_flows || []).map((flow: any, index: number) => ({
          id: flow.id || `edge_${index}`,
          source: flow.source,
          target: flow.target,
          type: mapRelationType(flow.type),
          weight: flow.weight || 0.5,
          frequency: flow.frequency || 1,
          lastActivity: flow.lastActivity || new Date().toISOString(),
          metadata: flow.metadata || {}
        }));

        // Generate communication patterns from relationships
        const transformedPatterns: CommunicationPattern[] = (networkData.relationships || [])
          .filter((rel: any) => rel.relationship_type === 'communication' || rel.relationship_type === 'contacted')
          .slice(0, 10)
          .map((rel: any, index: number) => ({
            participants: [rel.source_entity_id, rel.target_entity_id].filter(Boolean),
            frequency: rel.frequency || 1,
            pattern: rel.confidence > 0.8 ? 'suspicious' : 'normal',
            timeRange: rel.time_range || 'Recent',
            suspicionScore: rel.confidence || 0.5
          }));

        setNodes(transformedNodes);
        setEdges(transformedEdges);
        setPatterns(transformedPatterns);
        
      } catch (error) {
        console.error('Error fetching network data:', error);
        setError(`Failed to load network data: ${error}`);
        
        // Fallback to sample data if API fails
        const fallbackNodes: NetworkNode[] = [
          {
            id: 'no_data',
            label: 'No Network Data Available',
            type: 'person',
            importance: 0.5,
            connections: 0,
            metadata: { error: 'Could not load network data from API' }
          }
        ];
        
        setNodes(fallbackNodes);
        setEdges([]);
        setPatterns([]);
      } finally {
        setLoading(false);
      }
    };

    fetchNetworkData();
  }, []);

  // Helper functions to map backend data types to frontend types
  const mapEntityType = (backendType: string): NetworkNode['type'] => {
    const typeMap: { [key: string]: NetworkNode['type'] } = {
      'person': 'person',
      'phone': 'phone',
      'phone_number': 'phone',
      'email': 'email',
      'location': 'location',
      'device': 'device',
      'account': 'account'
    };
    return typeMap[backendType?.toLowerCase()] || 'person';
  };

  const mapRelationType = (backendType: string): NetworkEdge['type'] => {
    const typeMap: { [key: string]: NetworkEdge['type'] } = {
      'communication': 'communication',
      'contacted': 'communication',
      'owns': 'ownership',
      'ownership': 'ownership',
      'located_at': 'location',
      'location': 'location',
      'transaction': 'transaction',
      'related': 'association',
      'association': 'association'
    };
    return typeMap[backendType?.toLowerCase()] || 'association';
  };

  const getNodeColor = (type: NetworkNode['type']) => {
    switch (type) {
      case 'person': return '#3B82F6';
      case 'phone': return '#10B981';
      case 'email': return '#F59E0B';
      case 'location': return '#EF4444';
      case 'device': return '#8B5CF6';
      case 'account': return '#06B6D4';
      default: return '#6B7280';
    }
  };

  const getNodeIcon = (type: NetworkNode['type']) => {
    switch (type) {
      case 'person': return <Users className="h-4 w-4" />;
      case 'phone': return <Phone className="h-4 w-4" />;
      case 'email': return <Mail className="h-4 w-4" />;
      case 'location': return <Target className="h-4 w-4" />;
      case 'device': return <Network className="h-4 w-4" />;
      case 'account': return <GitBranch className="h-4 w-4" />;
      default: return <Network className="h-4 w-4" />;
    }
  };

  const getPatternBadgeColor = (pattern: CommunicationPattern['pattern']) => {
    switch (pattern) {
      case 'suspicious': return 'destructive';
      case 'burst': return 'default';
      case 'regular': return 'secondary';
      case 'normal': return 'outline';
      default: return 'secondary';
    }
  };

  const filteredNodes = nodes.filter(node =>
    node.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const refreshNetworkData = async () => {
    setLoading(true);
    // Trigger the useEffect by changing a dependency or call fetchNetworkData directly
    window.location.reload(); // Simple refresh for now
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
        <div className="text-center">
          <h2 className="text-xl font-semibold">Loading Network Analysis</h2>
          <p className="text-muted-foreground">Analyzing forensic data and building relationship graph...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Network Analysis</h1>
          <p className="text-muted-foreground">Interactive relationship mapping and communication pattern analysis</p>
          {currentCase && (
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="outline">Case: {currentCase}</Badge>
              <Badge variant="secondary">{nodes.length} Entities</Badge>
              <Badge variant="secondary">{edges.length} Connections</Badge>
            </div>
          )}
          {error && (
            <div className="mt-2 text-sm text-destructive">
              {error}
            </div>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={refreshNetworkData}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Refresh Data
          </Button>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Graph
          </Button>
          <Button variant="outline" size="sm">
            <Share2 className="h-4 w-4 mr-2" />
            Share Analysis
          </Button>
        </div>
      </div>

      {/* Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4 items-center flex-wrap">
            <div className="relative flex-1 min-w-64">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search entities in network..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Time Range:</span>
              <div className="w-32">
                <Slider
                  value={timeRange}
                  onValueChange={setTimeRange}
                  max={100}
                  step={1}
                  className="w-full"
                />
              </div>
            </div>
            <Button variant="outline" size="sm">
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Layout
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Mode Tabs */}
      <Tabs value={analysisMode} onValueChange={(value) => setAnalysisMode(value as any)}>
        <TabsList>
          <TabsTrigger value="static">Static Analysis</TabsTrigger>
          <TabsTrigger value="temporal">Temporal Evolution</TabsTrigger>
          <TabsTrigger value="simulation">AI Simulation</TabsTrigger>
        </TabsList>

        <TabsContent value="static" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Main Network Visualization */}
            <div className="lg:col-span-3">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Network className="h-5 w-5" />
                    Network Graph
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="relative bg-muted/20 rounded-lg" style={{ height: '500px' }}>
                    <canvas 
                      ref={canvasRef}
                      className="w-full h-full rounded-lg"
                      style={{ background: 'radial-gradient(circle, #1a1a1a 0%, #000000 100%)' }}
                    />
                    {/* Placeholder for network visualization */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <Network className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                        <h3 className="font-medium mb-2">Interactive Network Graph</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                          3D force-directed graph showing entity relationships
                        </p>
                        <Button>
                          <Brain className="h-4 w-4 mr-2" />
                          Generate Network
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Sidebar Controls and Info */}
            <div className="space-y-4">
              {/* Node Details */}
              {selectedNode ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {getNodeIcon(selectedNode.type)}
                      Node Details
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2">{selectedNode.label}</h4>
                      <div className="grid grid-cols-1 gap-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Type:</span>
                          <Badge variant="secondary" className="capitalize">
                            {selectedNode.type}
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Importance:</span>
                          <span>{Math.round(selectedNode.importance * 100)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Connections:</span>
                          <span>{selectedNode.connections}</span>
                        </div>
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h5 className="font-medium mb-2">Metadata</h5>
                      <div className="space-y-1 text-sm">
                        {Object.entries(selectedNode.metadata).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-muted-foreground capitalize">{key}:</span>
                            <span>{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="text-center py-8">
                    <Target className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">
                      Select a node to view details
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Network Statistics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Network Stats
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Total Nodes:</span>
                    <span className="font-medium">{nodes.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Total Edges:</span>
                    <span className="font-medium">{edges.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Density:</span>
                    <span className="font-medium">
                      {nodes.length > 1 ? Math.round((edges.length / (nodes.length * (nodes.length - 1) / 2)) * 100) : 0}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Avg. Connections:</span>
                    <span className="font-medium">
                      {nodes.length > 0 ? Math.round(nodes.reduce((sum, node) => sum + node.connections, 0) / nodes.length) : 0}
                    </span>
                  </div>
                </CardContent>
              </Card>

              {/* Top Entities */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="h-5 w-5" />
                    Key Entities
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-32">
                    <div className="space-y-2">
                      {nodes
                        .sort((a, b) => b.importance - a.importance)
                        .slice(0, 5)
                        .map((node) => (
                          <div key={node.id} className="flex items-center gap-2 text-sm">
                            {getNodeIcon(node.type)}
                            <span className="flex-1 truncate">{node.label}</span>
                            <Badge 
                              variant="secondary" 
                              className="text-xs"
                              style={{ backgroundColor: getNodeColor(node.type) + '20' }}
                            >
                              {Math.round(node.importance * 100)}%
                            </Badge>
                          </div>
                        ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Communication Patterns */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Communication Patterns
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {patterns.map((pattern, index) => (
                  <Card key={index} className="border">
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant={getPatternBadgeColor(pattern.pattern)}>
                          {pattern.pattern}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {pattern.frequency} interactions
                        </span>
                      </div>
                      <div className="space-y-1 text-sm">
                        <div>
                          <span className="text-muted-foreground">Participants: </span>
                          <span>{pattern.participants.join(', ')}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Period: </span>
                          <span>{pattern.timeRange}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Suspicion Score: </span>
                          <div className="flex-1">
                            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                              <div 
                                className={`h-full transition-all ${
                                  pattern.suspicionScore > 0.7 ? 'bg-red-500' :
                                  pattern.suspicionScore > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                                }`}
                                style={{ width: `${pattern.suspicionScore * 100}%` }}
                              />
                            </div>
                          </div>
                          <span className="text-xs">{Math.round(pattern.suspicionScore * 100)}%</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="temporal" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Temporal Network Evolution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Button
                    variant={isPlaying ? "default" : "outline"}
                    size="sm"
                    onClick={() => setIsPlaying(!isPlaying)}
                  >
                    {isPlaying ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                    {isPlaying ? 'Pause' : 'Play'} Timeline
                  </Button>
                  <Button variant="outline" size="sm">
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset
                  </Button>
                  <div className="flex-1">
                    <Slider
                      value={timeRange}
                      onValueChange={setTimeRange}
                      max={100}
                      step={1}
                      className="w-full"
                    />
                  </div>
                </div>
                
                <div className="h-96 bg-muted/20 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <Clock className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                    <h3 className="font-medium mb-2">Temporal Network Animation</h3>
                    <p className="text-sm text-muted-foreground">
                      Watch how relationships evolve over time
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="simulation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                AI Network Simulation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Button className="h-20 flex-col">
                    <Brain className="h-6 w-6 mb-2" />
                    <span>Predict Missing Links</span>
                  </Button>
                  <Button variant="outline" className="h-20 flex-col">
                    <TrendingUp className="h-6 w-6 mb-2" />
                    <span>Influence Analysis</span>
                  </Button>
                  <Button variant="outline" className="h-20 flex-col">
                    <Network className="h-6 w-6 mb-2" />
                    <span>Community Detection</span>
                  </Button>
                </div>
                
                <div className="h-96 bg-muted/20 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <Zap className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                    <h3 className="font-medium mb-2">AI-Powered Simulation</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Run intelligent scenarios and predictions on your network
                    </p>
                    <Button>
                      <Brain className="h-4 w-4 mr-2" />
                      Start Simulation
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default NetworkAnalysis;