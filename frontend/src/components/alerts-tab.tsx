"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { toast } from "sonner";
import { Trash2 } from "lucide-react";
import { SymbolSwitcher } from "./symbol-switcher";

interface Alert {
  id: number;
  name: string;
  symbol: string;
  conditions: any;
  notes?: string;
  is_active: boolean;
  created_at: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888";

export function AlertsTab({ symbol, onSymbolChange }: { symbol: string; onSymbolChange: (newSymbol: string) => void; }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [alertName, setAlertName] = useState("");
  const [alertSymbol, setAlertSymbol] = useState(symbol);
  const [alertMetric, setAlertMetric] = useState("price.close");
  const [alertOperator, setAlertOperator] = useState("GREATER_THAN");
  const [alertValue, setAlertValue] = useState("");
  const [alertNotes, setAlertNotes] = useState("");

  useEffect(() => {
    setAlertSymbol(symbol);
  }, [symbol]);

  const fetchAlerts = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_URL}/api/alerts/`);
      setAlerts(response.data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch alerts.");
      toast.error("Failed to fetch alerts.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  const handleDelete = async (id: number) => {
    try {
      await axios.delete(`${API_URL}/api/alerts/${id}`);
      toast.success("Alert deleted successfully!");
      fetchAlerts(); // Refresh list
    } catch (err) {
      toast.error("Failed to delete alert.");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!alertName || !alertSymbol || !alertMetric || !alertOperator || !alertValue) {
      toast.error("Please fill out all required fields.");
      return;
    }

    const newAlert = {
      name: alertName,
      symbol: alertSymbol.toUpperCase(),
      conditions: {
        logical_operator: "AND",
        conditions: [
          {
            metric: alertMetric,
            operator: alertOperator,
            value: parseFloat(alertValue),
          },
        ],
      },
      notes: alertNotes,
      is_active: true,
    };

    try {
      await axios.post(`${API_URL}/api/alerts/`, newAlert);
      toast.success("Alert created successfully!");
      fetchAlerts(); // Refresh list
      // Reset form
      setAlertName("");
      setAlertMetric("price.close");
      setAlertOperator("GREATER_THAN");
      setAlertValue("");
      setAlertNotes("");
    } catch (err) {
      toast.error("Failed to create alert.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Create New Alert</CardTitle>
          <CardDescription>
            Set up custom alerts to be notified of specific market conditions.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="space-y-2">
              <Label htmlFor="alertName">Alert Name</Label>
              <Input
                id="alertName"
                placeholder="e.g., 'AAPL Volatility Spike'"
                value={alertName}
                onChange={(e) => setAlertName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="alertSymbol">Symbol</Label>
              <Input
                id="alertSymbol"
                placeholder="e.g., 'AAPL'"
                value={alertSymbol}
                onChange={(e) => setAlertSymbol(e.target.value.toUpperCase())}
              />
            </div>
             <div className="space-y-2">
              <Label htmlFor="alertMetric">Metric</Label>
               <Select value={alertMetric} onValueChange={setAlertMetric}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a metric" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="price.close">Price (Close)</SelectItem>
                  <SelectItem value="volatility.predicted">Predicted Volatility</SelectItem>
                  <SelectItem value="sentiment.score">Sentiment Score</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="alertOperator">Operator</Label>
              <Select value={alertOperator} onValueChange={setAlertOperator}>
                <SelectTrigger>
                  <SelectValue placeholder="Select an operator" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="GREATER_THAN">Greater Than</SelectItem>
                  <SelectItem value="LESS_THAN">Less Than</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="alertValue">Value</Label>
              <Input
                id="alertValue"
                type="number"
                placeholder="e.g., 150.50"
                value={alertValue}
                onChange={(e) => setAlertValue(e.target.value)}
              />
            </div>
             <div className="space-y-2">
              <Label htmlFor="alertNotes">Notes (Optional)</Label>
              <Input
                id="alertNotes"
                placeholder="e.g., 'Watch for breakout'"
                value={alertNotes}
                onChange={(e) => setAlertNotes(e.target.value)}
              />
            </div>
          </CardContent>
          <CardFooter>
            <Button type="submit">Create Alert</Button>
          </CardFooter>
        </form>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Your Alerts</CardTitle>
          <CardDescription>A list of all your configured alerts across all symbols.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p>Loading alerts...</p>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Condition</TableHead>
                  <TableHead>Created At</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {alerts.length > 0 ? (
                  alerts.map((alert) => (
                    <TableRow key={alert.id}>
                      <TableCell className="font-medium">{alert.name}</TableCell>
                      <TableCell>
                        <Button variant="link" className="p-0 h-auto" onClick={() => onSymbolChange(alert.symbol)}>
                          {alert.symbol}
                        </Button>
                      </TableCell>
                      <TableCell>
                        {`${alert.conditions.conditions[0].metric} ${alert.conditions.conditions[0].operator.replace(/_/g, " ")} ${alert.conditions.conditions[0].value}`}
                      </TableCell>
                      <TableCell>
                        {new Date(alert.created_at).toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDelete(alert.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center">
                      No active alerts found.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 