"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SearchSubmitAnimation } from "../LoadingStates";
import IntentVisualizer from "./IntentVisualizer";
import { ChevronDown } from "lucide-react";

interface Props {
  onSearch: (query: string) => void;
  isSearching: boolean;
}

const EXAMPLES = [
  "Comedy films with Tom Hanks from the 90s",
  "Movies that feel like Blade Runner 2049",
  "A heist film where the twist is the crew planned to fail",
];

const FILTERS = ["Genre", "Decade", "Language", "Min Rating"];

export default function SearchInputPanel({ onSearch, isSearching }: Props) {
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [query]);

  // Detected Intent logic
  const getSimulatedIntent = () => {
    if (query.length < 5) return null;
    if (query.toLowerCase().includes("like") || query.toLowerCase().includes("similar")) return "RECOMMENDATION";
    if (/\b(?:with|by|starring|directed)\b/i.test(query)) return "ACTOR/DIRECTOR SEARCH";
    if (query.toLowerCase().includes("and") || query.toLowerCase().includes("from the")) return "HYBRID QUERY";
    return "PLOT SEARCH";
  };
  const intent = getSimulatedIntent();

  const handleSearchClick = () => {
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <div className="w-full h-full min-h-[100dvh] bg-carbon flex flex-col relative pt-8 px-8 pb-12 overflow-y-auto custom-scrollbar">
      
      {/* Sticky header */}
      <div className="sticky top-0 bg-carbon z-20 pb-4 pt-2">
        <div className="flex items-baseline">
          <span className="font-display text-2xl tracking-wide text-white">PlotSense</span>
          <span className="w-[5px] h-[5px] rounded-full bg-aqua ml-1 mb-[3px]" />
        </div>
      </div>

      <div className="flex-1 mt-12 w-full max-w-[500px] mx-auto flex flex-col">
        
        {/* Main Textarea */}
        <div className="relative w-full">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Describe the movie you're thinking of..."
            className="w-full bg-transparent font-sans text-[20px] text-white placeholder-steel/50 outline-none resize-none overflow-hidden min-h-[60px]"
            rows={1}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSearchClick();
              }
            }}
          />
          
          {/* Glowing underline */}
          <motion.div
            layout
            initial={false}
            animate={{
              boxShadow: isFocused ? "0 0 12px rgba(132, 220, 198, 0.2)" : "none",
              borderColor: isFocused ? "#84DCC6" : "rgba(75, 78, 109, 1)",
              borderWidth: isFocused ? 2 : 1
            }}
            className="absolute bottom-0 left-0 right-0 h-0 border-b border-grape transition-colors duration-200"
          />

          {/* Char count & real-time intent */}
          <div className="absolute -bottom-6 left-0 right-0 flex justify-between items-center">
             <span className="font-mono text-[11px] text-steel/60">
               {query.length} chars
             </span>

             <AnimatePresence mode="wait">
               {intent && (
                 <motion.div
                   key={intent}
                   initial={{ scale: 0.85, opacity: 0 }}
                   animate={{ scale: 1, opacity: 1 }}
                   exit={{ scale: 0.85, opacity: 0 }}
                   className="bg-grape text-mono text-[11px] text-white px-2 py-[2px] rounded-full"
                 >
                   {intent === "RECOMMENDATION" ? "🎞️" : intent === "ACTOR/DIRECTOR SEARCH" ? "🎭" : intent === "HYBRID QUERY" ? "✨" : "🔍"} {intent}
                 </motion.div>
               )}
             </AnimatePresence>
          </div>
        </div>

        {/* Intent Component (Mounts at 10 chars) */}
        <div className="mt-8 mb-4 min-h-[140px]">
          {query.length >= 10 && (
             <IntentVisualizer isAnalyzing={query.length >= 10} hasSubmitted={isSearching} />
          )}

          {/* Examples (When empty) */}
          <AnimatePresence>
            {query.length === 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col gap-2 mt-4"
              >
                {EXAMPLES.map((ex) => (
                  <motion.button
                    key={ex}
                    onClick={() => { setQuery(ex); setTimeout(() => textareaRef.current?.focus(), 50); }}
                    whileHover={{ borderColor: "#84DCC6", color: "#FFFFFF", backgroundColor: "rgba(75, 78, 109, 0.2)" }}
                    className="self-start text-left bg-transparent border border-grape text-steel font-sans text-[13px] rounded-full px-4 py-1.5 transition-colors"
                  >
                    {ex}
                  </motion.button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Search Button */}
        <div className="mt-4" onClick={handleSearchClick}>
          <SearchSubmitAnimation isSubmitting={isSearching} />
        </div>

        {/* Filters */}
        <div className="mt-6">
           <button 
             onClick={() => setShowFilters(!showFilters)}
             className="flex items-center text-steel font-sans text-sm hover:text-white transition-colors"
           >
             Filters <ChevronDown className={`ml-1 w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
           </button>
           <AnimatePresence>
             {showFilters && (
               <motion.div
                 initial={{ height: 0, opacity: 0 }}
                 animate={{ height: "auto", opacity: 1 }}
                 exit={{ height: 0, opacity: 0 }}
                 className="overflow-hidden flex flex-wrap gap-2 pt-3"
               >
                  {FILTERS.map(f => (
                    <button key={f} className="font-sans text-[12px] bg-carbon border border-grape text-steel px-3 py-1 rounded-full hover:border-aqua hover:text-aqua transition-colors">
                      {f} ▾
                    </button>
                  ))}
               </motion.div>
             )}
           </AnimatePresence>
        </div>

      </div>

      {/* Keyboard Shortcuts Bar */}
      <div className="mt-auto pt-10">
        <div className="font-mono text-[11px] text-steel/40">
          ⌘+K New search &nbsp;·&nbsp; ↑↓ Navigate &nbsp;·&nbsp; Enter Open
        </div>
      </div>

    </div>
  );
}
