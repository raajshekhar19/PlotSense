"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SkeletonResultCard } from "../LoadingStates";
import AnswerSynthesis from "./AnswerSynthesis";

interface MatchResult {
  id: string;
  title: string;
  year: string;
  genre: string;
  snippet: string;
  score: number;
  sources: string[];
  director: string;
  posterUrl?: string; // Optional real TMDB url
}

interface Props {
  query: string;
  intentHover: string | null;
  results: MatchResult[] | null;
  isLoading: boolean;
  aiAnswer?: string | null;
  onResultClick: (result: MatchResult) => void;
}

export default function ResultsStream({ query, intentHover, results, isLoading, aiAnswer, onResultClick }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  
  // Highlight snippet Helper
  const renderSnippet = (text: string) => {
    // Simple mock highlighting
    return text.split(/(<mark>.*?<\/mark>)/g).map((part, i) => {
      if (part.startsWith("<mark>") && part.endsWith("</mark>")) {
        return (
          <span key={i} className="bg-aqua/20 text-aqua rounded-[3px] px-[2px]">
            {part.replace(/<\/?mark>/g, "")}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  // SVG Ring calculation
  const getScoreColor = (score: number) => {
    if (score >= 90) return "#84DCC6";
    if (score >= 70) return "#95A3B3";
    return "#4B4E6D";
  };

  return (
    <div className="w-full h-full min-h-[100dvh] bg-[#1A1A1A] py-8 px-8 overflow-y-auto custom-scrollbar flex flex-col relative border-l border-grape/30">
        
        {/* Results Header */}
        {(isLoading || results) && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-between w-full mb-6"
          >
             <div className="flex items-center gap-4">
               <span className="font-sans text-[13px] text-steel/70 italic">
                 Results for → {query || "..."}
               </span>
               <div className="bg-grape text-aqua font-mono text-[11px] px-2 py-0.5 rounded-full uppercase tracking-wider">
                 {intentHover || "PLOT SEARCH"}
               </div>
               {results && (
                 <span className="font-sans text-[13px] text-steel">
                   {results.length} matches found
                 </span>
               )}
             </div>
             
             <div className="text-[13px] font-sans text-steel cursor-pointer hover:text-white transition-colors">
               Relevance ▾
             </div>
          </motion.div>
        )}

        {/* Answer Synthesis (Phase 2 integration - if results exist) */}
        {results && results.length > 0 && !isLoading && aiAnswer && (
          <div className="mb-8">
            <AnswerSynthesis answerText={aiAnswer} />
          </div>
        )}

        {/* Skeletons Layout */}
        {isLoading && (
          <div className="flex flex-col gap-4">
            {[1, 2, 3, 4].map((i, idx) => (
               <SkeletonResultCard key={i} staggerDelay={idx * 0.1} />
            ))}
          </div>
        )}

        {/* Results Layout */}
        {!isLoading && results && results.length > 0 && (
          <div className="flex flex-col gap-4 pb-20">
            {results.map((r, idx) => {
              const isBestMatch = idx === 0;
              const radius = 24;
              const circumference = 2 * Math.PI * radius;
              const offset = circumference - (r.score / 100) * circumference;

              return (
                <motion.div
                  layout
                  key={r.id}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30, delay: idx * 0.08 }}
                  onClick={() => {
                    setExpandedId(expandedId === r.id ? null : r.id);
                    onResultClick(r);
                  }}
                  className={`relative w-full bg-[#2A2A2A] rounded-2xl p-5 flex gap-5 cursor-pointer font-sans transition-all group overflow-hidden ${
                    isBestMatch 
                      ? "border-[1.5px] border-aqua/50 scale-[1.01]" 
                      : "border border-grape/40 hover:border-aqua/60 hover:bg-[#2E2E2E]"
                  }`}
                  whileHover={!isBestMatch ? { scale: 1.005 } : undefined}
                >
                  {/* Hover shimmer effect */}
                  <motion.div 
                    initial={{ x: "-100%", opacity: 0 }}
                    whileHover={{ x: "200%", opacity: 1 }}
                    transition={{ duration: 1, ease: "easeInOut" }}
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-aqua/5 to-transparent skew-x-12 pointer-events-none"
                  />

                  {/* Best Match Badge */}
                  {isBestMatch && (
                    <div className="absolute top-0 right-4 -translate-y-1/2 rotate-[-2deg] font-mono text-[10px] bg-aqua text-carbon font-bold px-2 py-0.5 rounded shadow-[0_4px_12px_rgba(132,220,198,0.3)]">
                      BEST MATCH
                    </div>
                  )}

                  {/* Poster Left */}
                  <div className="w-[80px] h-[110px] bg-grape/30 rounded-xl shrink-0 flex items-center justify-center overflow-hidden border border-grape border-dashed">
                    {r.posterUrl ? (
                      <img src={r.posterUrl} alt={r.title} className="w-full h-full object-cover" />
                    ) : (
                      <div className="text-aqua/40">
                         {/* Film Icon Placeholder */}
                         <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>
                      </div>
                    )}
                  </div>

                  {/* Center Content */}
                  <div className="flex-1 flex flex-col justify-center">
                    <h3 className="text-white font-bold text-[18px] leading-tight flex items-center gap-2">
                       {r.title} 
                       <span className="font-normal text-[13px] text-steel font-mono">{r.year} &nbsp;·&nbsp; {r.genre}</span>
                    </h3>
                    
                    <motion.p 
                      layout="position"
                      className={`mt-2 text-[14px] text-steel/80 leading-relaxed ${expandedId === r.id ? "" : "line-clamp-2"}`}
                    >
                      {renderSnippet(r.snippet)}
                    </motion.p>
                  </div>

                  {/* Right Meta (Score ring) */}
                  <div className="w-[80px] shrink-0 flex flex-col items-center justify-center gap-2 border-l border-grape/20 pl-4 py-1 relative">
                     <div className="relative w-[56px] h-[56px] flex items-center justify-center">
                        {/* Background ring */}
                        <svg className="absolute w-full h-full -rotate-90">
                           <circle cx="28" cy="28" r={radius} fill="none" stroke="rgba(75,78,109,0.3)" strokeWidth="4" />
                           <motion.circle 
                             cx="28" cy="28" r={radius}
                             fill="none" 
                             stroke={getScoreColor(r.score)} 
                             strokeWidth="4" 
                             strokeDasharray={circumference}
                             initial={{ strokeDashoffset: circumference }}
                             animate={{ strokeDashoffset: offset }}
                             transition={{ duration: 1.5, ease: "easeOut", delay: idx * 0.1 + 0.3 }}
                             strokeLinecap="round"
                           />
                        </svg>
                        <span className="font-mono text-white text-[14px] font-bold">{r.score}%</span>
                     </div>

                     <div className="flex flex-col gap-1 items-center mt-1">
                       {r.sources.map(s => (
                          <div key={s} className={`text-[9px] font-mono px-1.5 py-[1px] rounded uppercase ${s === 'FAISS' ? 'bg-grape/40 text-steel' : 'bg-aqua/20 text-aqua border border-aqua/20'}`}>
                            {s}
                          </div>
                       ))}
                     </div>

                     {/* View Details hover text */}
                     <span className="absolute -bottom-2 right-0 left-4 text-center text-aqua text-[12px] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                       {expandedId === r.id ? "Minimize" : "View Details →"}
                     </span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}

        {/* Empty State */}
        {!isLoading && results && results.length === 0 && (
           <motion.div 
             initial={{ opacity: 0 }} animate={{ opacity: 1 }}
             className="w-full flex-1 flex flex-col items-center justify-center text-center -mt-20"
           >
              {/* Illustration lines */}
              <div className="mb-6 opacity-60">
                <svg width="120" height="120" viewBox="0 0 120 120" fill="none" stroke="#4B4E6D" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                   <circle cx="60" cy="60" r="50"/>
                   <circle cx="60" cy="60" r="16"/>
                   <line x1="60" y1="10" x2="60" y2="44"/>
                   <line x1="60" y1="76" x2="60" y2="110"/>
                   <line x1="10" y1="60" x2="44" y2="60"/>
                   <line x1="76" y1="60" x2="110" y2="60"/>
                   <circle cx="35" cy="35" r="4"/>
                   <circle cx="85" cy="35" r="4"/>
                   <circle cx="35" cy="85" r="4"/>
                   <circle cx="85" cy="85" r="4"/>
                </svg>
              </div>
              <p className="font-sans text-[16px] text-steel mb-8">
                Nothing found. Try describing the vibe differently.
              </p>
              <div className="flex gap-4">
                 <button className="text-[13px] bg-transparent border border-grape text-steel px-4 py-2 rounded-lg hover:border-aqua hover:text-white transition-colors">
                   "Visually stunning sci-fi"
                 </button>
                 <button className="text-[13px] bg-transparent border border-grape text-steel px-4 py-2 rounded-lg hover:border-aqua hover:text-white transition-colors">
                   "Movies directed by Nolan"
                 </button>
              </div>
           </motion.div>
        )}
    </div>
  );
}
