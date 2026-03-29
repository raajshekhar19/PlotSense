"use client";

import { useState } from "react";
import Nav from "@/components/Nav";
import SearchInputPanel from "@/components/search/SearchInputPanel";
import ResultsStream from "@/components/search/ResultsStream";
import { ProgressBar } from "@/components/LoadingStates";
import { motion, AnimatePresence } from "framer-motion";

// Mock Results generator
const MOCK_RESULTS = [
  {
    id: "1",
    title: "Inception",
    year: "2010",
    genre: "Sci-Fi / Thriller",
    snippet: "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the <mark>psychological heist</mark>.",
    score: 98,
    sources: ["FAISS", "Neo4j", "Reranker"],
    director: "Christopher Nolan",
    posterUrl: "https://image.tmdb.org/t/p/w200/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg"
  },
  {
    id: "2",
    title: "Shutter Island",
    year: "2010",
    genre: "Thriller / Mystery",
    snippet: "In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane. It shares similar deeply <mark>psychological unreliable narrator</mark> elements.",
    score: 87,
    sources: ["FAISS"],
    director: "Martin Scorsese",
    posterUrl: "https://image.tmdb.org/t/p/w200/qayga0775aEaZ2yXyCjN3uNWe8o.jpg"
  },
  {
    id: "3",
    title: "The Prestige",
    year: "2006",
    genre: "Drama / Mystery / Sci-Fi",
    snippet: "After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other. Fits the <mark>twisty narrative</mark>.",
    score: 65,
    sources: ["Neo4j"],
    director: "Christopher Nolan",
    posterUrl: "https://image.tmdb.org/t/p/w200/jZKcOAlXQ10O2Vq81Hl5N4DlsxL.jpg"
  }
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<typeof MOCK_RESULTS | null>(null);

  const handleSearch = (newQuery: string) => {
    setQuery(newQuery);
    setIsSearching(true);
    setResults(null); // Clear old results
    
    // Simulate pipeline loading time
    setTimeout(() => {
      setResults(MOCK_RESULTS);
      setIsSearching(false);
    }, 2500);
  };

  return (
    <div className="flex flex-col min-h-screen bg-carbon overflow-hidden">
      <Nav />
      
      {/* Search Progress Bar (Top) */}
      <div className="fixed top-[60px] left-0 right-0 z-40">
        {(isSearching || results) && (
          <ProgressBar progress={isSearching ? null : 100} />
        )}
      </div>

      <div className="flex flex-col md:flex-row flex-1 pt-[60px]">
        {/* LEFT PANEL */}
        <div className="w-full md:w-[40%] flex-shrink-0 h-[calc(100vh-60px)] relative border-r border-grape/30 z-10">
          <SearchInputPanel 
            onSearch={handleSearch} 
            isSearching={isSearching} 
          />
        </div>

        {/* RIGHT PANEL */}
        <div className="w-full md:w-[60%] h-[calc(100vh-60px)] relative">
          <AnimatePresence mode="popLayout">
            {/* Desktop always shows ResultsStream. Mobile relies on logic below */}
            <div className="hidden md:block w-full h-full">
              <ResultsStream 
                query={query}
                intentHover={null}
                results={results}
                isLoading={isSearching}
                onResultClick={(r) => console.log(r)} 
              />
            </div>
            
            {/* Mobile slide up drawer for results */}
            {(isSearching || results) && (
              <motion.div
                initial={{ y: "100%" }}
                animate={{ y: "0%" }}
                exit={{ y: "100%" }}
                transition={{ type: "spring", stiffness: 350, damping: 40 }}
                className="md:hidden absolute inset-0 z-50 bg-[#1A1A1A] rounded-t-3xl shadow-2xl overflow-hidden shadow-[0_-10px_40px_rgba(0,0,0,0.5)]"
              >
                {/* Mobile Handle */}
                <div className="w-full flex justify-center pt-3 pb-1">
                  <div className="w-12 h-1.5 bg-grape/50 rounded-full" />
                </div>
                <ResultsStream 
                  query={query}
                  intentHover={null}
                  results={results}
                  isLoading={isSearching}
                  onResultClick={(r) => console.log(r)} 
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
