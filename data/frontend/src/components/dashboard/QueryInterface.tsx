import { useState } from "react";
import { Search, Send, Mic, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

const sampleQueries = [
  "Show me chat records containing crypto addresses",
  "List all communications with foreign numbers",
  "Find deleted messages from the last 30 days",
  "Show me all contacts with suspicious activity",
  "Extract GPS locations from media files",
];

const recentQueries = [
  { query: "Show me WhatsApp messages mentioning 'bitcoin'", results: 23, timestamp: "2 hours ago" },
  { query: "Find all calls to international numbers", results: 156, timestamp: "4 hours ago" },
  { query: "Extract deleted photos from device", results: 89, timestamp: "1 day ago" },
];

export function QueryInterface() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string>("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setErrorMsg("");
    try {
      const response = await fetch("http://localhost:5000/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });
      if (response.ok) {
        // You can handle the response data here if needed
        setQuery("");
      } else {
        setErrorMsg("Query failed. Please try again.");
      }
    } catch (error) {
      setErrorMsg("Error connecting to backend.");
    }
    setIsLoading(false);
  };

  const handleSampleQuery = (sampleQuery: string) => {
    setQuery(sampleQuery);
  };

  return (
    <Card className="card-professional">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="w-5 h-5 text-primary" />
          Natural Language Query Interface
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
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
                  <p className="text-xs text-muted-foreground">{item.timestamp}</p>
                </div>
                <Badge variant="secondary" className="ml-2">
                  {item.results} results
                </Badge>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}