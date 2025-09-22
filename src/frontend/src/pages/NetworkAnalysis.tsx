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
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Mock data - replace with API calls to your knowledge graph backend
  useEffect(() => {
    const mockNodes: NetworkNode[] = [
      {
        id: 'person1',
        label: 'John Doe',
        type: 'person',
        importance: 0.95,
        connections: 8,
        metadata: { role: 'suspect', verified: true }
      },
      {
        id: 'phone1',
        label: '+1-555-0123',
        type: 'phone',
        importance: 0.85,
        connections: 5,
        metadata: { carrier: 'Verizon', type: 'mobile' }
      },
      {
        id: 'person2',
        label: 'Jane Smith',
        type: 'person',
        importance: 0.70,
        connections: 4,
        metadata: { role: 'associate', verified: false }
      },
      {
        id: 'email1',
        label: 'suspect@email.com',
        type: 'email',
        importance: 0.80,
        connections: 6,
        metadata: { provider: 'Gmail', verified: true }
      },
      {
        id: 'location1',
        label: 'Downtown Bank',
        type: 'location',
        importance: 0.65,
        connections: 3,
        metadata: { address: '123 Main St', type: 'financial' }
      }
    ];

    const mockEdges: NetworkEdge[] = [
      {
        id: 'edge1',
        source: 'person1',
        target: 'phone1',
        type: 'ownership',
        weight: 0.95,
        frequency: 1,
        lastActivity: '2024-01-15T14:30:00Z',
        metadata: { verified: true }
      },
      {
        id: 'edge2',
        source: 'person1',
        target: 'person2',
        type: 'communication',
        weight: 0.75,
        frequency: 23,
        lastActivity: '2024-01-15T12:00:00Z',
        metadata: { method: 'calls', suspicious: true }
      },
      {
        id: 'edge3',
        source: 'phone1',
        target: 'location1',
        type: 'location',
        weight: 0.60,
        frequency: 5,
        lastActivity: '2024-01-14T10:45:00Z',
        metadata: { duration: '2 hours' }
      }
    ];

    const mockPatterns: CommunicationPattern[] = [
      {
        participants: ['John Doe', 'Jane Smith'],
        frequency: 23,
        pattern: 'suspicious',
        timeRange: 'Jan 10-15, 2024',
        suspicionScore: 0.85
      },
      {
        participants: ['John Doe', '+1-555-0123'],
        frequency: 1,
        pattern: 'normal',
        timeRange: 'Jan 15, 2024',
        suspicionScore: 0.20
      }
    ];

    setTimeout(() => {
      setNodes(mockNodes);
      setEdges(mockEdges);
      setPatterns(mockPatterns);
      setLoading(false);
    }, 1000);
  }, []);

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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
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