"use client";

import * as React from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { useSymbolStore } from "@/lib/store";

import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

const symbols = [
  { value: "AAPL", label: "Apple Inc." },
  { value: "NVDA", label: "NVIDIA Corp." },
  { value: "TSLA", label: "Tesla, Inc." },
  { value: "MSFT", label: "Microsoft Corp." },
  { value: "GOOGL", label: "Alphabet Inc." },
  { value: "AMZN", label: "Amazon.com, Inc." },
];

export function SymbolSwitcher() {
  const [open, setOpen] = React.useState(false);
  const { currentSymbol, setCurrentSymbol } = useSymbolStore();

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-[200px] justify-between"
        >
          {currentSymbol
            ? symbols.find((symbol) => symbol.value === currentSymbol)?.label
            : "Select symbol..."}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandInput placeholder="Search symbol..." />
          <CommandList>
            <CommandEmpty>No symbol found.</CommandEmpty>
            <CommandGroup>
              {symbols.map((symbol) => (
                <CommandItem
                  key={symbol.value}
                  value={symbol.value}
                  onSelect={(currentValue) => {
                    setCurrentSymbol(currentValue.toUpperCase());
                    setOpen(false);
                  }}
                >
                  <Check
                    className={`mr-2 h-4 w-4 ${
                      currentSymbol === symbol.value ? "opacity-100" : "opacity-0"
                    }`}
                  />
                  {symbol.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
} 