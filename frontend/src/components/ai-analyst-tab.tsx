"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Lightbulb, TrendingUp, Zap } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

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
  symbol: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888";

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

export function AIAnalystTab({ symbol }: AIAnalystTabProps) {
  const [data, setData] = useState<AIAnalystData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_URL}/api/insights/${symbol}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch AI analysis: ${response.statusText}`);
        }
        const result: AIAnalystData = await response.json();
        setData(result);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <Skeleton className="h-8 w-3/4" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6 mt-2" />
          </CardContent>
        </Card>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[...Array(3)].map((_, index) => (
            <Card key={index}>
              <CardHeader>
                <Skeleton className="h-6 w-4/5" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-6 w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }
  
  if (error) {
    return <div className="text-red-500">Error: {error}</div>;
  }

  if (!data || data.insights.length === 0) {
    return <div>No AI analysis available for {symbol}.</div>;
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