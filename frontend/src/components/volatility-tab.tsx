"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import apiClient from "@/lib/api";
import { toast } from "sonner";
import { useSymbolStore } from "@/lib/store";
import { Skeleton } from "./ui/skeleton";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888";

interface VolatilityData {
  predicted_volatility: number;
  prediction_range: {
    low: number;
    high: number;
  };
  volatility_regime: string;
  trend: string;
  confidence_score: number;
}

interface HistoricalVolatility {
  date: string;
  value: number;
}

interface VolatilityTabProps {
  symbol: string;
}

export function VolatilityTab({ symbol }: VolatilityTabProps) {
  const [volatilityData, setVolatilityData] = useState<VolatilityData | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalVolatility[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const [volatilityResponse, historicalResponse] = await Promise.all([
          fetch(`${API_URL}/api/volatility/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: symbol })
          }),
          fetch(`${API_URL}/api/volatility/historical/${symbol}`)
        ]);

        if (!volatilityResponse.ok) {
          throw new Error(`Failed to fetch volatility forecast: ${volatilityResponse.statusText}`);
        }
        if (!historicalResponse.ok) {
          throw new Error(`Failed to fetch historical volatility: ${historicalResponse.statusText}`);
        }

        const volData = await volatilityResponse.json();
        const histData = await historicalResponse.json();

        setVolatilityData(volData);
        setHistoricalData(histData);

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
          <CardContent className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex justify-between">
                <Skeleton className="h-5 w-1/3" />
                <Skeleton className="h-5 w-1/4" />
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-4 w-1/3" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[250px] w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return <div className="text-red-500">Error: {error}</div>;
  }

  if (!volatilityData) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-dashed shadow-sm h-96">
        <div className="text-center text-muted-foreground">
          Volatility data not available for {symbol}.
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Volatility Forecast</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Predicted Volatility</span>
            <strong>{volatilityData.predicted_volatility?.toFixed(4)}</strong>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Prediction Range</span>
            <strong>
              {volatilityData.prediction_range?.low?.toFixed(4)} -{" "}
              {volatilityData.prediction_range?.high?.toFixed(4)}
            </strong>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Volatility Regime</span>
            <strong className="capitalize">{volatilityData.volatility_regime}</strong>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Trend</span>
            <strong className="capitalize">{volatilityData.trend}</strong>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Confidence Score</span>
            <strong>{(volatilityData.confidence_score * 100)?.toFixed(1)}%</strong>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Historical Volatility</CardTitle>
          <CardDescription>Last 90 days</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            {historicalData && historicalData.length > 0 ? (
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  stroke="#ccc"
                  tickFormatter={(str) => new Date(str).toLocaleDateString("en-US", { month: "short", day: "numeric"})}
                />
                <YAxis stroke="#ccc" domain={['dataMin - 0.01', 'dataMax + 0.01']} />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: "hsl(var(--background))",
                    borderColor: "hsl(var(--border))",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#8884d8"
                  activeDot={{ r: 8 }}
                  dot={false}
                />
              </LineChart>
            ) : (
              <div className="flex h-full w-full items-center justify-center">
                <p className="text-muted-foreground">No historical data available.</p>
              </div>
            )}
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
} 