import { LucideIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

interface StatsCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    label: string;
  };
  className?: string;
}

export function StatsCard({ title, value, description, icon: Icon, trend, className = "" }: StatsCardProps) {
  return (
    <Card className={`card-professional hover-glow ${className}`}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <div className="flex items-baseline gap-2 mt-2">
              <h3 className="text-2xl font-bold text-foreground">{value}</h3>
              {trend && (
                <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                  trend.value > 0 
                    ? 'bg-success/20 text-success' 
                    : trend.value < 0 
                    ? 'bg-destructive/20 text-destructive' 
                    : 'bg-muted/20 text-muted-foreground'
                }`}>
                  {trend.value > 0 ? '+' : ''}{trend.value}% {trend.label}
                </span>
              )}
            </div>
            {description && (
              <p className="text-xs text-muted-foreground mt-1">{description}</p>
            )}
          </div>
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
            <Icon className="w-6 h-6 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}