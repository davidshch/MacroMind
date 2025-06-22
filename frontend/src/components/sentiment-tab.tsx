"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Skeleton } from "./ui/skeleton";
import { Badge } from "./ui/badge";

interface SentimentData {
  news_sentiment_details: any;
  reddit_sentiment_details: any;
  news_sentiment_score: number;
  reddit_sentiment_score: number;
  [key: string]: any; // Allow other properties
}

interface SentimentTabProps {
  symbol: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888";

export function SentimentTab({ symbol }: SentimentTabProps) {
  const [data, setData] = useState<SentimentData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_URL}/api/sentiment/${symbol}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch sentiment data: ${response.statusText}`);
        }
        const result: SentimentData = await response.json();
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
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-4 w-1/3" />
          </CardHeader>
          <CardContent>
             <Skeleton className="h-40 w-full" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-4 w-1/3" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-40 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }
  
  if (error) {
    return <div className="text-red-500">Error: {error}</div>;
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-dashed shadow-sm h-96">
        <div className="text-center text-muted-foreground">
          Sentiment data not available for {symbol}.
        </div>
      </div>
    );
  }

  const {
    news_sentiment_details,
    reddit_sentiment_details,
    news_sentiment_score,
    reddit_sentiment_score,
  } = data;

  const getSentimentBadgeVariant = (sentiment: string) => {
    switch (sentiment?.toLowerCase()) {
      case "bullish":
        return "default";
      case "bearish":
        return "destructive";
      default:
        return "secondary";
    }
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>News Sentiment</CardTitle>
          <CardDescription>
            Overall Score: {news_sentiment_score?.toFixed(3)}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Top Articles</TableHead>
                <TableHead className="text-right">Sentiment</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {news_sentiment_details?.top_articles?.map((article: any, index: number) => (
                <TableRow key={index}>
                  <TableCell>
                    <a
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium hover:underline"
                    >
                      {article.title}
                    </a>
                    <p className="text-sm text-muted-foreground">{article.source}</p>
                  </TableCell>
                  <TableCell className="text-right capitalize">
                     <Badge variant={getSentimentBadgeVariant(article.sentiment)}>
                      {article.sentiment}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Social Sentiment (Reddit)</CardTitle>
          <CardDescription>
            Overall Score: {reddit_sentiment_score?.toFixed(3)}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Subreddit</TableHead>
                <TableHead>Posts</TableHead>
                <TableHead className="text-right">Engagement</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {reddit_sentiment_details?.subreddit_breakdown &&
                Object.entries(
                  reddit_sentiment_details.subreddit_breakdown
                ).map(([subreddit, details]: [string, any]) => (
                  <TableRow key={subreddit}>
                    <TableCell className="font-medium">{`r/${subreddit}`}</TableCell>
                    <TableCell>{details.post_count}</TableCell>
                    <TableCell className="text-right">
                      {details.total_engagement}
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
} 