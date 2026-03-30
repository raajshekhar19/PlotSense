"use client";

import { useState } from "react";
import Nav from "@/components/Nav";
import SearchInputPanel from "@/components/search/SearchInputPanel";
import ResultsStream from "@/components/search/ResultsStream";
import { ProgressBar } from "@/components/LoadingStates";
import { motion, AnimatePresence } from "framer-motion";

const BACKEND_URL = "http://localhost:8000";

interface BackendResponse {
  query: string;
  intent: string | null;
  movie_name: string | null;
  answer: string;
  kg_movies: string[] | null;
  needs_clarification: boolean | null;
  clarification_question: string | null;
}

interface MatchResult {
  id: string;
  title: string;
  year: string;
  genre: string;
  snippet: string;
  score: number;
  sources: string[];
  director: string;
  posterUrl?: string;
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<MatchResult[] | null>(null);
  const [intent, setIntent] = useState<string | null>(null);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (newQuery: string) => {
    setQuery(newQuery);
    setIsSearching(true);
    setResults(null);
    setAiAnswer(null);
    setError(null);

    try {
      const res = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: newQuery }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Backend returned ${res.status}`);
      }

      const data: BackendResponse = await res.json();

      setIntent(data.intent);
      setAiAnswer(data.answer);

      // ── Clarification short-circuit ───────────────────────────
      // When the backend doesn't recognise the movie, surface the
      // clarification question as a single info card and stop here.
      if (data.needs_clarification && data.clarification_question) {
        setResults([
          {
            id: "clarification-0",
            title: "🤔 Hmm, I'm not sure about that one…",
            year: "",
            genre: "clarification",
            snippet: data.clarification_question,
            score: 0,
            sources: ["PlotSense"],
            director: "",
          },
        ]);
        return;
      }

      // Build result cards from the backend response
      const cards: MatchResult[] = [];

      // If kg_movies were returned, create cards for each
      if (data.kg_movies && data.kg_movies.length > 0) {
        data.kg_movies.forEach((movie: any, idx: number) => {
          if (typeof movie === "string") {
            cards.push({
              id: `kg-${idx}`,
              title: movie,
              year: "",
              genre: data.intent || "search",
              snippet: data.answer || "",
              score: Math.max(60, 98 - idx * 8),
              sources: data.intent === "plot" ? ["FAISS"] : ["Neo4j"],
              director: "",
            });
          } else if (typeof movie === "object" && movie !== null) {
            cards.push({
              id: `kg-${idx}`,
              title: movie.title || "Unknown",
              year: movie.year || "",
              genre: data.intent || "search",
              snippet: movie.snippet || data.answer || "",
              score: Math.max(60, 98 - idx * 8),
              sources: movie.source ? [movie.source] : (data.intent === "plot" ? ["FAISS"] : ["Neo4j"]),
              director: movie.director || "",
              posterUrl: movie.posterUrl || "",
            });
          }
        });
      }

      // If the intent found a specific movie name (like querying for recommendations),
      // we don't want the movie itself showing up in the results, so remove it if it exists
      if (data.movie_name) {
        const movieIndex = cards.findIndex(c => c.title.toLowerCase() === data.movie_name!.toLowerCase());
        if (movieIndex !== -1) {
          cards.splice(movieIndex, 1);
        }
      }

      // If we got an answer but no structured results, create a single result card from the answer
      if (cards.length === 0 && data.answer && data.answer.length > 10) {
        // Try to extract movie titles from the AI answer
        const titleMatches = data.answer.match(/[""]([^""]+)[""]|"([^"]+)"/g);
        if (titleMatches && titleMatches.length > 0) {
          titleMatches.slice(0, 5).forEach((match, idx) => {
            const title = match.replace(/["""]/g, "").trim();
            if (title.length > 2 && title.length < 80) {
              cards.push({
                id: `extracted-${idx}`,
                title: title,
                year: "",
                genre: data.intent || "",
                snippet: data.answer || "",
                score: Math.max(55, 95 - idx * 10),
                sources: data.intent === "plot" ? ["FAISS"] : ["Neo4j"],
                director: "",
              });
            }
          });
        }
      }
      
      // Fallback: show the raw AI answer as a single card if nothing else
      if (cards.length === 0 && data.answer) {
        cards.push({
          id: "answer-0",
          title: data.movie_name || "AI Response",
          year: "",
          genre: data.intent || "search",
          snippet: data.answer || "",
          score: 90,
          sources: ["LLM"],
          director: "",
        });
      }

      setResults(cards);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      console.error("Search failed:", message);
      setError(message);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
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
          {/* Show errors */}
          {error && (
            <div className="absolute bottom-20 left-8 right-8 bg-red-900/30 border border-red-500/50 text-red-300 text-sm p-3 rounded-xl font-sans">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* RIGHT PANEL */}
        <div className="w-full md:w-[60%] h-[calc(100vh-60px)] relative">
          {/* Desktop */}
          <div className="hidden md:block w-full h-full" key="desktop-results">
            <ResultsStream 
              query={query}
              intentHover={intent}
              results={results}
              isLoading={isSearching}
              aiAnswer={aiAnswer}
              onResultClick={(r) => console.log("Clicked:", r)} 
            />
          </div>
          
          {/* Mobile slide up drawer for results */}
          <AnimatePresence>
            {(isSearching || results) && (
              <motion.div
                key="mobile-results-drawer"
                initial={{ y: "100%" }}
                animate={{ y: "0%" }}
                exit={{ y: "100%" }}
                transition={{ type: "spring", stiffness: 350, damping: 40 }}
                className="md:hidden absolute inset-0 z-50 bg-[#1A1A1A] rounded-t-3xl shadow-2xl overflow-hidden"
              >
                <div className="w-full flex justify-center pt-3 pb-1">
                  <div className="w-12 h-1.5 bg-grape/50 rounded-full" />
                </div>
                <ResultsStream 
                  query={query}
                  intentHover={intent}
                  results={results}
                  isLoading={isSearching}
                  aiAnswer={aiAnswer}
                  onResultClick={(r) => console.log("Clicked:", r)} 
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
