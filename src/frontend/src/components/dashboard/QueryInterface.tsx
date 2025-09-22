import { useState, useEffect } from "react";
import { Search, Send, Mic, Loader2, MessageSquare } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";

const sampleQueries = [
  "Show me chat records containing crypto addresses",
  "List all communications with foreign numbers", 
  "Find deleted messages from the last 30 days",
  "Show me all contacts with suspicious activity",
  "Extract GPS locations from media files",
  "Analyze call patterns and frequency",
  "Find evidence of data tampering",
  "Search for financial transaction keywords"
];

interface QueryResult {
  query: string;
  results: number;
  timestamp: string;
  confidence: number;
}

interface CaseInfo {
  id: string;
  processed_evidence_count: number;
  total_evidence_count: number;
}

export function QueryInterface() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [recentQueries, setRecentQueries] = useState<QueryResult[]>([]);
  const [caseInfo, setCaseInfo] = useState<CaseInfo | null>(null);
  const { toast } = useToast();
  const navigate = useNavigate();

  useEffect(() => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (selectedCaseId) {
      fetchCaseInfo(selectedCaseId);
      loadRecentQueries(selectedCaseId);
    }
  }, []);

  const fetchCaseInfo = async (caseId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/cases/${caseId}`);
      if (response.ok) {
        const data = await response.json();
        setCaseInfo(data.case);
      }
    } catch (error) {
      console.error('Error fetching case info:', error);
    }
  };

  const loadRecentQueries = (caseId: string) => {
    // Load recent queries from localStorage for this case
    const storageKey = `recentQueries_${caseId}`;
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      try {
        setRecentQueries(JSON.parse(stored));
      } catch (error) {
        console.error('Error loading recent queries:', error);
      }
    }
  };

  const saveRecentQuery = (queryText: string, results: number, confidence: number) => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) return;

    const storageKey = `recentQueries_${selectedCaseId}`;
    const newQuery: QueryResult = {
      query: queryText,
      results,
      timestamp: new Date().toISOString(),
      confidence
    };

    const updated = [newQuery, ...recentQueries.slice(0, 4)]; // Keep only last 5
    setRecentQueries(updated);
    localStorage.setItem(storageKey, JSON.stringify(updated));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      toast({
        title: "Error",
        description: "No case selected",
        variant: "destructive"
      });
      return;
    }

    if (!caseInfo || caseInfo.processed_evidence_count === 0) {
      toast({
        title: "No Evidence Processed",
        description: "Please upload and process evidence files before querying",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    setErrorMsg("");
    
    try {
      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: query,
          case_id: selectedCaseId,
          conversation_history: []
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const resultsCount = data.sources ? data.sources.length : 0;
        
        // Save this query to recent queries
        saveRecentQuery(query, resultsCount, data.confidence || 0);
        
        // Show success message
        toast({
          title: "Query Executed",
          description: `Found ${resultsCount} relevant evidence sources. Go to Chat for detailed analysis.`,
        });
        
        setQuery("");
        
        // Optionally navigate to chat to see results
        navigate('/query');
        
      } else {
        const errorData = await response.json();
        setErrorMsg(errorData.detail || "Query failed. Please try again.");
      }
    } catch (error) {
      setErrorMsg("Error connecting to backend. Please check if the API server is running.");
      console.error('Error executing query:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleQuery = (sampleQuery: string) => {
    setQuery(sampleQuery);
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) {
      const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
      return diffInMinutes < 1 ? "Just now" : `${diffInMinutes} minutes ago`;
    } else if (diffInHours < 24) {
      return `${diffInHours} hours ago`;
    } else {
      const diffInDays = Math.floor(diffInHours / 24);
      return `${diffInDays} days ago`;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const canQuery = caseInfo && caseInfo.processed_evidence_count > 0;

  return (
    <Card className="card-professional">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Search className="w-5 h-5 text-primary" />
            Natural Language Query Interface
          </div>
          {caseInfo && (
            <div className="flex items-center gap-2">
              <Badge variant={canQuery ? "default" : "destructive"}>
                {caseInfo.processed_evidence_count}/{caseInfo.total_evidence_count} Files Ready
              </Badge>
              {canQuery && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigate('/query')}
                  className="text-xs"
                >
                  <MessageSquare className="w-3 h-3 mr-1" />
                  Full Chat
                </Button>
              )}
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {!canQuery ? (
          <div className="text-center py-8 bg-muted/30 rounded-lg">
            <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="font-medium text-lg mb-2">Evidence Processing Required</h3>
            <p className="text-muted-foreground mb-4">
              Upload and process evidence files to enable natural language queries
            </p>
            <div className="text-sm text-muted-foreground">
              {caseInfo ? (
                `${caseInfo.processed_evidence_count} of ${caseInfo.total_evidence_count} files processed`
              ) : (
                "No case selected"
              )}
            </div>
          </div>
        ) : (
          <>
            {/* Main Query Input */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Type your investigation query in natural language..."
                  className="pl-10 pr-20 h-12 text-base bg-secondary/50 border-border/50 focus:bg-secondary/70"
                  disabled={isLoading}
                />
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="w-8 h-8 p-0"
                    disabled={isLoading}
                  >
                    <Mic className="w-4 h-4" />
                  </Button>
                  <Button
                    type="submit"
                    variant="professional"
                    size="sm"
                    disabled={!query.trim() || isLoading}
                  >
                    {isLoading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                  </Button>
                </div>
              </div>
              {errorMsg && (
                <div className="mt-2 text-red-600 text-sm font-medium">{errorMsg}</div>
              )}
            </form>

            {/* Sample Queries */}
            <div>
              <h4 className="text-sm font-medium text-muted-foreground mb-3">Sample Queries</h4>
              <div className="flex flex-wrap gap-2">
                {sampleQueries.map((sample, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => handleSampleQuery(sample)}
                    className="text-xs text-left h-auto py-2 px-3 whitespace-normal hover:bg-primary/10 hover:border-primary/30"
                    disabled={isLoading}
                  >
                    {sample}
                  </Button>
                ))}
              </div>
            </div>

            {/* Recent Queries */}
            {recentQueries.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-3">Recent Queries</h4>
                <div className="space-y-2">
                  {recentQueries.map((item, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg hover:bg-secondary/50 transition-colors cursor-pointer"
                      onClick={() => handleSampleQuery(item.query)}
                    >
                      <div className="flex-1">
                        <p className="text-sm font-medium text-foreground">{item.query}</p>
                        <p className="text-xs text-muted-foreground">{formatTimestamp(item.timestamp)}</p>
                      </div>
                      <div className="flex items-center gap-2 ml-2">
                        <Badge variant="secondary">
                          {item.results} results
                        </Badge>
                        {item.confidence > 0 && (
                          <Badge variant="outline" className={`text-xs ${getConfidenceColor(item.confidence)}`}>
                            {(item.confidence * 100).toFixed(0)}%
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}