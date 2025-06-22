"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Lightbulb, TrendingUp, Zap } from "lucide-react";

interface Insight {
  title: string;
  description: string;
  confidence: number;
  category: string;
}

interface AIAnalystData {
  symbol: string;
  generated_at: string;
  overall_outlook: string;
  summary: string;
  insights: Insight[];
}

interface AIAnalystTabProps {
  data: AIAnalystData | null;
  isLoading: boolean;
}

const getCategoryIcon = (category: string) => {
  switch (category.toLowerCase()) {
    case "volatility":
      return <Zap className="h-5 w-5 text-yellow-500" />;
    case "sentiment":
      return <TrendingUp className="h-5 w-5 text-blue-500" />;
    case "opportunity":
        return <Lightbulb className="h-5 w-5 text-green-500" />;
    default:
      return <Lightbulb className="h-5 w-5 text-gray-500" />;
  }
};

export function AIAnalystTab({ data, isLoading }: AIAnalystTabProps) {
  if (isLoading) {
    return <div>Loading AI analysis...</div>;
  }

  if (!data) {
    return <div>No AI analysis available.</div>;
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>AI Analyst Outlook: {data.overall_outlook}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">{data.summary}</p>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {data.insights.map((insight, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{insight.title}</CardTitle>
              {getCategoryIcon(insight.category)}
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground">{insight.description}</p>
              <Badge variant="outline" className="mt-2">
                Confidence: {(insight.confidence * 100).toFixed(0)}%
              </Badge>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
} 