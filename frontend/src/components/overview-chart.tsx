"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Skeleton } from "./ui/skeleton";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888";

interface OverviewChartProps {
  symbol: string;
}

export function OverviewChart({ symbol }: OverviewChartProps) {
  const [data, setData] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_URL}/api/visualization/historical-prices/${symbol}?days=90`);
        if (!response.ok) {
          throw new Error(`Failed to fetch price data: ${response.statusText}`);
        }
        const result = await response.json();
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
    return <Skeleton className="h-[350px] w-full" />;
  }
  
  if (error) {
    return <p className="text-red-500">Error loading chart data: {error}</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={350}>
      {data && data.length > 0 ? (
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="date"
            stroke="#ccc"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
          />
          <YAxis
            yAxisId="left"
            orientation="left"
            stroke="#ccc"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `$${value}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--background))",
              borderColor: "hsl(var(--border))",
            }}
          />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="price"
            stroke="#8884d8"
            name="Price"
            dot={false}
          />
        </LineChart>
      ) : (
        <p>No data available for {symbol}</p>
      )}
    </ResponsiveContainer>
  );
}