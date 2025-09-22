import { 
  FileText, 
  Search, 
  Upload, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Users,
  Database,
  TrendingUp
} from "lucide-react";
import { useState } from "react";
import { StatsCard } from "@/components/dashboard/StatsCard";
import { RecentActivity } from "@/components/dashboard/RecentActivity";
import { QueryInterface } from "@/components/dashboard/QueryInterface";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import forensicHero from "@/assets/forensic-hero.jpg";

const Dashboard = () => {
    const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle");
    const [uploadMessage, setUploadMessage] = useState<string>("");
      const handleUpload = async (file: File) => {
        const formData = new FormData();
        formData.append("ufdr", file);

        setUploadStatus("idle");
        setUploadMessage("");

        try {
          const response = await fetch("http://localhost:5000/api/upload", {
            method: "POST",
            body: formData,
          });
          if (response.ok) {
            setUploadStatus("success");
            setUploadMessage("File uploaded successfully!");
          } else {
            setUploadStatus("error");
            setUploadMessage("Upload failed. Please try again.");
          }
        } catch (error) {
          setUploadStatus("error");
          setUploadMessage("Error uploading file. Please check your connection.");
        }
      };
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-subtle">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-20"
          style={{ backgroundImage: `url(${forensicHero})` }}
        />
        <div className="relative p-8">
          <div className="max-w-2xl">
            <h1 className="text-4xl font-bold mb-4">
              Welcome to ForensicAI
            </h1>
            <p className="text-xl text-muted-foreground mb-6">
              Advanced AI-powered digital forensic investigation platform. Extract actionable insights from UFDRs with natural language queries.
            </p>
            <div className="flex gap-4">
            <Button variant="professional" size="lg" asChild>
              <label className="cursor-pointer">
                <Upload className="mr-2 w-5 h-5" />
                Upload UFDR
                <input
                  type="file"
                  className="hidden"
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handleUpload(file);
                    }
                  }}
                />
              </label>
            </Button>
              <Button variant="investigation" size="lg">
                <Search className="mr-2 w-5 h-5" />
                Start Investigation
              </Button>
            </div>
            {/* Success/Error UI */}
            {uploadStatus === "success" && (
              <div className="mt-4 text-green-600 font-medium">{uploadMessage}</div>
            )}
            {uploadStatus === "error" && (
              <div className="mt-4 text-red-600 font-medium">{uploadMessage}</div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Active Cases"
          value={23}
          description="Currently under investigation"
          icon={FileText}
          trend={{ value: 12, label: "this month" }}
        />
        <StatsCard
          title="Evidence Processed"
          value="1.2TB"
          description="Total data analyzed"
          icon={Database}
          trend={{ value: 8, label: "this week" }}
        />
        <StatsCard
          title="Queries Executed"
          value={456}
          description="Natural language searches"
          icon={Search}
          trend={{ value: 23, label: "today" }}
        />
        <StatsCard
          title="Reports Generated"
          value={89}
          description="Investigation summaries"
          icon={TrendingUp}
          trend={{ value: 15, label: "this month" }}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Query Interface */}
        <div className="lg:col-span-2">
          <QueryInterface />
        </div>

        {/* Quick Actions & Status */}
        <div className="space-y-6">
          {/* System Status */}
          <Card className="card-professional">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-success" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">AI Processing Engine</span>
                <Badge className="status-active">Online</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Database Connection</span>
                <Badge className="status-active">Connected</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Evidence Storage</span>
                <Badge className="status-active">Available</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Report Generator</span>
                <Badge className="status-pending">Processing</Badge>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="card-professional">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button variant="outline" className="w-full justify-start" size="sm">
                <Upload className="mr-2 w-4 h-4" />
                Upload New UFDR
              </Button>
              <Button variant="outline" className="w-full justify-start" size="sm">
                <FileText className="mr-2 w-4 h-4" />
                Generate Report
              </Button>
              <Button variant="outline" className="w-full justify-start" size="sm">
                <Users className="mr-2 w-4 h-4" />
                Case Management
              </Button>
              <Button variant="outline" className="w-full justify-start" size="sm">
                <AlertTriangle className="mr-2 w-4 h-4" />
                Security Alerts
              </Button>
            </CardContent>
          </Card>

          {/* Alerts */}
          <Card className="card-professional border-destructive/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                <AlertTriangle className="w-5 h-5" />
                Security Alerts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-3 bg-destructive/10 rounded-lg border border-destructive/20">
                <p className="text-sm font-medium text-destructive">Unauthorized Access Attempt</p>
                <p className="text-xs text-muted-foreground">Failed login from IP: 192.168.1.100</p>
                <p className="text-xs text-muted-foreground">5 minutes ago</p>
              </div>
              <div className="p-3 bg-warning/10 rounded-lg border border-warning/20">
                <p className="text-sm font-medium text-warning">High CPU Usage</p>
                <p className="text-xs text-muted-foreground">Processing engine at 89% capacity</p>
                <p className="text-xs text-muted-foreground">15 minutes ago</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Recent Activity */}
      <RecentActivity />
    </div>
  );
};

export default Dashboard;