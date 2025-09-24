import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Search, Database, FileText, MessageSquare, Filter, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface SearchResult {
  case_id?: string;
  case_title?: string;
  source_file?: string;
  file_type?: string;
  content?: string;
  timestamp?: string;
  relevance_score?: number;
  highlights?: string[];
  metadata?: any;
}

interface CaseSummary {
  case_id: string;
  title: string;
  description?: string;
  status: string;
  created_at?: string;
  summary?: any;
}

const DatabaseSearch = () => {
  const [query, setQuery] = useState('');
  const [selectedCase, setSelectedCase] = useState<string>('');
  const [dataType, setDataType] = useState<string>('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalResults, setTotalResults] = useState(0);
  const { toast } = useToast();

  // Fetch available cases on component mount
  useEffect(() => {
    fetchCases();
  }, []);

  const fetchCases = async () => {
    try {
      const response = await fetch('http://localhost:8000/cases');
      const data = await response.json();
      
      if (Array.isArray(data)) {
        // Transform the data to match our interface
        const transformedCases = data.map(case_ => ({
          case_id: case_.id || case_.case_id,
          title: case_.title,
          description: case_.description,
          status: case_.status,
          created_at: case_.created_at
        }));
        setCases(transformedCases);
      } else {
        toast({
          title: "Error",
          description: "Failed to load cases",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error fetching cases:', error);
      toast({
        title: "Error",
        description: "Failed to connect to the database",
        variant: "destructive",
      });
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: "Warning",
        description: "Please enter a search query",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const params = new URLSearchParams({
        query: query.trim(),
        max_results: '20'
      });

      if (selectedCase && selectedCase !== 'all') {
        params.append('case_id', selectedCase);
      }

      if (dataType && dataType !== 'all') {
        params.append('data_type', dataType);
      }

      const response = await fetch(`http://localhost:8000/database/search?${params}`);
      const data = await response.json();

      if (data.success) {
        setResults(data.results);
        setTotalResults(data.total_results);
        toast({
          title: "Search Complete",
          description: `Found ${data.total_results} results`,
        });
      } else {
        toast({
          title: "Search Failed",
          description: "No results found or search error occurred",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Search error:', error);
      toast({
        title: "Error",
        description: "Failed to perform search",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const clearFilters = () => {
    setSelectedCase('all');
    setDataType('all');
    setQuery('');
    setResults([]);
    setTotalResults(0);
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  const truncateContent = (content: string, maxLength: number = 200) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Database className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Database Search</h1>
          <p className="text-muted-foreground">Search through forensic case data and evidence</p>
        </div>
      </div>

      {/* Search Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Search Parameters
          </CardTitle>
          <CardDescription>
            Configure your search query and filters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search Query */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Search Query</label>
            <div className="flex gap-2">
              <Input
                placeholder="Enter search terms..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                className="flex-1"
              />
              <Button onClick={handleSearch} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                Search
              </Button>
            </div>
          </div>

          {/* Filters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Case Filter</label>
              <Select value={selectedCase} onValueChange={setSelectedCase}>
                <SelectTrigger>
                  <SelectValue placeholder="All cases" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All cases</SelectItem>
                  {cases.map((case_) => (
                    <SelectItem key={case_.case_id} value={case_.case_id}>
                      {case_.title || `Case ${case_.case_id}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Data Type</label>
              <Select value={dataType} onValueChange={setDataType}>
                <SelectTrigger>
                  <SelectValue placeholder="All types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  <SelectItem value="messages">Messages</SelectItem>
                  <SelectItem value="files">Files</SelectItem>
                  <SelectItem value="contacts">Contacts</SelectItem>
                  <SelectItem value="locations">Locations</SelectItem>
                  <SelectItem value="metadata">Metadata</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Actions</label>
              <Button variant="outline" onClick={clearFilters} className="w-full">
                <Filter className="h-4 w-4 mr-2" />
                Clear Filters
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Summary */}
      {totalResults > 0 && (
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">
                  Found <span className="font-semibold">{totalResults}</span> results
                  {selectedCase && selectedCase !== 'all' && (
                    <span> in case <Badge variant="secondary">{cases.find(c => c.case_id === selectedCase)?.title || selectedCase}</Badge></span>
                  )}
                </p>
              </div>
              <Badge variant="outline">
                Query: "{query}"
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Search Results */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <Card key={index} className="hover:shadow-md transition-shadow">
            <CardContent className="pt-6">
              <div className="space-y-3">
                {/* Result Header */}
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">
                        {result.source_file || 'Unknown Source'}
                      </span>
                      {result.file_type && (
                        <Badge variant="secondary" className="text-xs">
                          {result.file_type}
                        </Badge>
                      )}
                    </div>
                    {result.case_title && (
                      <div className="flex items-center gap-2">
                        <Database className="h-3 w-3 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">
                          {result.case_title}
                        </span>
                      </div>
                    )}
                  </div>
                  {result.relevance_score && (
                    <Badge variant={result.relevance_score > 0.7 ? "default" : "outline"}>
                      {(result.relevance_score * 100).toFixed(0)}% match
                    </Badge>
                  )}
                </div>

                <Separator />

                {/* Content */}
                {result.content && (
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">Content:</p>
                    <p className="text-sm bg-muted p-3 rounded-md">
                      {truncateContent(result.content)}
                    </p>
                  </div>
                )}

                {/* Highlights */}
                {result.highlights && result.highlights.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">Highlights:</p>
                    <div className="flex flex-wrap gap-2">
                      {result.highlights.slice(0, 3).map((highlight, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {truncateContent(highlight, 50)}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Metadata */}
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    {result.timestamp && `Timestamp: ${formatTimestamp(result.timestamp)}`}
                  </span>
                  {result.case_id && (
                    <span>Case ID: {result.case_id}</span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Empty State */}
      {!loading && results.length === 0 && query && (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-12">
              <Search className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Results Found</h3>
              <p className="text-muted-foreground mb-4">
                Try adjusting your search query or filters
              </p>
              <Button variant="outline" onClick={clearFilters}>
                Clear All Filters
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Initial State */}
      {!loading && results.length === 0 && !query && (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-12">
              <Database className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Database Search</h3>
              <p className="text-muted-foreground mb-4">
                Search through forensic case data, evidence, and messages
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                  <MessageSquare className="h-5 w-5 text-primary" />
                  <span className="text-sm">Messages</span>
                </div>
                <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                  <FileText className="h-5 w-5 text-primary" />
                  <span className="text-sm">Files</span>
                </div>
                <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                  <Database className="h-5 w-5 text-primary" />
                  <span className="text-sm">Metadata</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default DatabaseSearch;