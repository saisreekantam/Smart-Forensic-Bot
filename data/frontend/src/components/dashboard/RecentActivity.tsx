import { Clock, FileText, Search, Upload, Users } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const activities = [
  {
    id: 1,
    type: "query",
    description: "Natural language query executed",
    details: "Show me chat records containing crypto addresses",
    user: "Det. Johnson",
    timestamp: "2 minutes ago",
    status: "completed",
    icon: Search,
  },
  {
    id: 2,
    type: "upload",
    description: "UFDR file uploaded",
    details: "smartphone_case_2024_001.ufdr (2.4 GB)",
    user: "Det. Smith",
    timestamp: "15 minutes ago",
    status: "processing",
    icon: Upload,
  },
  {
    id: 3,
    type: "report",
    description: "Investigation report generated",
    details: "Case #2024-INV-001 - Financial Fraud",
    user: "Analyst Davis",
    timestamp: "1 hour ago",
    status: "completed",
    icon: FileText,
  },
  {
    id: 4,
    type: "query",
    description: "Network analysis completed",
    details: "Communications with foreign numbers",
    user: "Det. Wilson",
    timestamp: "2 hours ago",
    status: "completed",
    icon: Users,
  },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case "completed":
      return "status-active";
    case "processing":
      return "status-pending";
    case "failed":
      return "status-alert";
    default:
      return "status-pending";
  }
};

export function RecentActivity() {
  return (
    <Card className="card-professional">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="w-5 h-5 text-primary" />
          Recent Activity
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {activities.map((activity) => {
          const IconComponent = activity.icon;
          return (
            <div key={activity.id} className="flex items-start gap-3 p-3 rounded-lg bg-secondary/50 hover:bg-secondary/70 transition-colors">
              <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0">
                <IconComponent className="w-4 h-4 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-sm font-medium text-foreground">{activity.description}</p>
                  <Badge className={getStatusColor(activity.status)} variant="outline">
                    {activity.status}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mb-1">{activity.details}</p>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>{activity.user}</span>
                  <span>•</span>
                  <span>{activity.timestamp}</span>
                </div>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}