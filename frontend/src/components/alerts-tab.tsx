"use client";

import { useState } from 'react';
import api from '@/lib/api';
import { toast } from 'sonner';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { MoreHorizontal, Loader2 } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from './ui/skeleton';

interface Alert {
  id: number;
  name: string;
  symbol: string;
  conditions: any;
  created_at: string;
  is_active: boolean;
  notes?: string;
}

interface AlertsTabProps {
  alerts: Alert[];
  isLoading: boolean;
  symbol: string;
  onAlertChange: () => void; // Callback to refresh data on parent
}

export function AlertsTab({ alerts, isLoading, symbol, onAlertChange }: AlertsTabProps) {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  // Form state for creation
  const [alertName, setAlertName] = useState("");
  const [alertMetric, setAlertMetric] = useState("price.close");
  const [alertOperator, setAlertOperator] = useState("GREATER_THAN");
  const [alertValue, setAlertValue] = useState("");
  const [alertNotes, setAlertNotes] = useState("");

  // Form state for editing
  const [editAlertName, setEditAlertName] = useState("");
  const [editAlertNotes, setEditAlertNotes] = useState("");

  const resetCreateForm = () => {
    setAlertName("");
    setAlertMetric("price.close");
    setAlertOperator("GREATER_THAN");
    setAlertValue("");
    setAlertNotes("");
  };

  const handleDelete = async (alertId: number) => {
    const toastId = toast.loading('Deleting alert...');
    try {
      await api.delete(`/alerts/${alertId}`);
      toast.success('Alert deleted successfully!', { id: toastId });
      onAlertChange(); // Refresh the list
    } catch (error) {
      toast.error('Failed to delete alert.', { id: toastId });
    }
  };

  const handleCreateAlert = async (e: React.FormEvent) => {
    e.preventDefault();
    const toastId = toast.loading("Creating alert...");
    try {
      const alertData = {
        name: alertName,
        symbol: symbol,
        notes: alertNotes,
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
      };
      await api.post("/alerts/", alertData);
      toast.success("Alert created successfully!", { id: toastId });
      onAlertChange(); // Refresh the list
      setIsCreateDialogOpen(false); // Close the dialog
    } catch (error) {
      toast.error("Failed to create alert.", { id: toastId });
    }
  };

  const handleEditClick = (alert: Alert) => {
    setSelectedAlert(alert);
    setEditAlertName(alert.name);
    setEditAlertNotes(alert.notes || "");
    setIsEditDialogOpen(true);
  };

  const handleUpdateAlert = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedAlert) return;

    const toastId = toast.loading("Updating alert...");
    try {
      const updateData = {
        name: editAlertName,
        notes: editAlertNotes,
      };
      await api.put(`/alerts/${selectedAlert.id}`, updateData);
      toast.success("Alert updated successfully!", { id: toastId });
      onAlertChange();
      setIsEditDialogOpen(false);
      setSelectedAlert(null);
    } catch (error) {
      toast.error("Failed to update alert.", { id: toastId });
    }
  };

  const filteredAlerts = alerts.filter(
    (alert) => alert.symbol === symbol
  );

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Your Alerts for {symbol}</CardTitle>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button onClick={resetCreateForm}>Create New Alert</Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Create New Alert</DialogTitle>
              <DialogDescription>
                Get notified when your conditions are met for {symbol}.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleCreateAlert}>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="name" className="text-right">
                    Name
                  </Label>
                  <Input
                    id="name"
                    value={alertName}
                    onChange={(e) => setAlertName(e.target.value)}
                    className="col-span-3"
                    placeholder="e.g., 'AAPL High Volatility'"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="metric" className="text-right">
                    Metric
                  </Label>
                  <Select
                    value={alertMetric}
                    onValueChange={setAlertMetric}
                  >
                    <SelectTrigger className="col-span-3">
                      <SelectValue placeholder="Select a metric" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="price.close">Price (Close)</SelectItem>
                      <SelectItem value="volatility.predicted">
                        Predicted Volatility
                      </SelectItem>
                      <SelectItem value="sentiment.score">
                        Sentiment Score
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="operator" className="text-right">
                    Operator
                  </Label>
                  <Select
                    value={alertOperator}
                    onValueChange={setAlertOperator}
                  >
                    <SelectTrigger className="col-span-3">
                      <SelectValue placeholder="Select an operator" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="GREATER_THAN">Is Greater Than</SelectItem>
                      <SelectItem value="LESS_THAN">Is Less Than</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="value" className="text-right">
                    Value
                  </Label>
                  <Input
                    id="value"
                    type="number"
                    value={alertValue}
                    onChange={(e) => setAlertValue(e.target.value)}
                    className="col-span-3"
                    placeholder="e.g., '150.50'"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="notes" className="text-right">
                    Notes
                  </Label>
                  <Input
                    id="notes"
                    value={alertNotes}
                    onChange={(e) => setAlertNotes(e.target.value)}
                    className="col-span-3"
                    placeholder="Optional notes"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type="submit">Create Alert</Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : filteredAlerts.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Condition</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>
                  <span className="sr-only">Actions</span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAlerts.map((alert) => (
                <TableRow key={alert.id}>
                  <TableCell className="font-medium">{alert.name}</TableCell>
                  <TableCell>
                    {`${alert.conditions.conditions[0].metric.replace('.', ' ')} ${alert.conditions.conditions[0].operator.replace('_', ' ')} ${alert.conditions.conditions[0].value}`}
                  </TableCell>
                  <TableCell>{alert.is_active ? 'Active' : 'Inactive'}</TableCell>
                  <TableCell>{new Date(alert.created_at).toLocaleDateString()}</TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button aria-haspopup="true" size="icon" variant="ghost">
                          <MoreHorizontal className="h-4 w-4" />
                          <span className="sr-only">Toggle menu</span>
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuLabel>Actions</DropdownMenuLabel>
                        <DropdownMenuItem onClick={() => handleEditClick(alert)}>Edit</DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleDelete(alert.id)}>Delete</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="text-center py-8">
            <p className="text-muted-foreground">You have no alerts for {symbol}.</p>
            <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
              <DialogTrigger asChild>
                <Button onClick={resetCreateForm} className="mt-4">Create One Now</Button>
              </DialogTrigger>
              {/* Dialog Content same as above */}
            </Dialog>
          </div>
        )}
      </CardContent>

      {/* Edit Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit Alert</DialogTitle>
              <DialogDescription>
                Update the details for your alert.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleUpdateAlert}>
            <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-name" className="text-right">
                    Name
                  </Label>
                  <Input
                    id="edit-name"
                    value={editAlertName}
                    onChange={(e) => setEditAlertName(e.target.value)}
                    className="col-span-3"
                  />
                </div>
                 <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-notes" className="text-right">
                    Notes
                  </Label>
                  <Input
                    id="edit-notes"
                    value={editAlertNotes}
                    onChange={(e) => setEditAlertNotes(e.target.value)}
                    className="col-span-3"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type="submit">Save Changes</Button>
              </DialogFooter>
            </form>
          </DialogContent>
      </Dialog>
    </Card>
  );
} 