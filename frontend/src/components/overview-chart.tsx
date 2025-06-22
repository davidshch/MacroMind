"use client";

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

export function OverviewChart({ data, isLoading }: { data: any[], isLoading: boolean }) {
  if (isLoading) {
    return <Skeleton className="h-[350px] w-full" />;
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
        <p>No data available</p>
      )}
    </ResponsiveContainer>
  );
}