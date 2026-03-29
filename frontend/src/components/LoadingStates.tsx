"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { Search } from "lucide-react";

// COMPONENT 1 — SkeletonResultCard
export function SkeletonResultCard({ staggerDelay = 0, className = "" }: { staggerDelay?: number; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 30, delay: staggerDelay }}
      className={`w-full bg-[#2A2A2A] border border-grape/40 rounded-2xl p-5 flex gap-5 ${className}`}
    >
      {/* Poster shimmer */}
      <div className="w-[80px] h-[110px] rounded-xl animate-shimmer shrink-0" />
      
      {/* Center content */}
      <div className="flex-1 flex flex-col pt-1">
        <div className="w-[180px] h-[18px] rounded animate-shimmer mb-2" />
        <div className="w-[120px] h-[13px] rounded animate-shimmer mb-6" />
        
        <div className="w-[260px] max-w-full h-[13px] rounded animate-shimmer mb-2" />
        <div className="w-[240px] max-w-[90%] h-[13px] rounded animate-shimmer mb-2" />
        <div className="w-[200px] max-w-[80%] h-[13px] rounded animate-shimmer" />
      </div>

      {/* Right match score */}
      <div className="w-[56px] h-[56px] rounded-full animate-shimmer shrink-0" />
    </motion.div>
  );
}

// COMPONENT 2 — SkeletonDrawer
export function SkeletonDrawer({ className = "" }: { className?: string }) {
  return (
    <div className={`w-full h-full flex flex-col ${className}`}>
      {/* Hero Banner Skeleton */}
      <div className="w-full h-[220px] animate-shimmer relative">
        {/* Poster Skeleton */}
        <div className="absolute left-6 -bottom-10 w-[120px] h-[170px] rounded-xl animate-shimmer bg-carbon border-4 border-carbon shadow-2xl" />
      </div>
      
      <div className="px-6 pt-16 pb-8">
        <div className="w-[240px] h-8 rounded-lg animate-shimmer mb-3" />
        <div className="w-[160px] h-[14px] rounded animate-shimmer mb-10" />

        {/* Match Analysis Skeleton */}
        <div className="w-full h-[100px] rounded-xl animate-shimmer mb-8" />

        {/* Cast Grid Skeleton */}
        <div className="grid grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="flex gap-3 items-center">
              <div className="w-12 h-12 rounded-full animate-shimmer shrink-0" />
              <div className="flex flex-col gap-2">
                <div className="w-24 h-3 rounded animate-shimmer" />
                <div className="w-16 h-2 rounded animate-shimmer" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// COMPONENT 3 — AIThinkingIndicator
const THINKING_MESSAGES = [
  "Analyzing semantic similarity...",
  "Traversing knowledge graph...",
  "Reranking results...",
  "Composing answer..."
];

export function AIThinkingIndicator({ className = "" }: { className?: string }) {
  const [msgIdx, setMsgIdx] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMsgIdx((prev) => (prev + 1) % THINKING_MESSAGES.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`flex flex-col items-center gap-3 ${className}`}>
      <div className="flex justify-center gap-[6px]">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut", delay: i * 0.15 }}
            className="w-2 h-2 rounded-full bg-aqua"
          />
        ))}
      </div>
      <div className="h-[20px] overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.p
            key={msgIdx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="text-steel italic font-sans text-sm m-0 leading-tight"
          >
            {THINKING_MESSAGES[msgIdx]}
          </motion.p>
        </AnimatePresence>
      </div>
    </div>
  );
}

// COMPONENT 4 — ProgressBar
export function ProgressBar({ progress = null, className = "" }: { progress?: number | null; className?: string }) {
  const isIndeterminate = progress === null;
  
  return (
    <div className={`w-full h-[2px] bg-transparent overflow-hidden relative ${className}`}>
      {isIndeterminate ? (
        <motion.div
          animate={{ x: ["-100%", "200%"] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="absolute left-0 top-0 h-full w-[30%] bg-aqua"
        />
      ) : (
        <motion.div
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="h-full bg-aqua"
        />
      )}
      <AnimatePresence>
        {progress === 100 && (
          <motion.div
            initial={{ opacity: 1 }}
            animate={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="absolute inset-0 bg-white"
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// COMPONENT 5 — IntentBadgeSkeleton
export function IntentBadgeSkeleton({ className = "" }: { className?: string }) {
  return (
    <motion.div 
      animate={{ opacity: [0.3, 0.7, 0.3] }}
      transition={{ duration: 1.2, repeat: Infinity }}
      className={`w-[100px] h-[24px] bg-grape/50 rounded-full ${className}`}
    />
  );
}

// COMPONENT 6 — SearchSubmitAnimation
export function SearchSubmitAnimation({ 
  isSubmitting, 
  onComplete 
}: { 
  isSubmitting: boolean; 
  onComplete?: () => void;
}) {
  const [phase, setPhase] = useState<"idle" | "loading" | "complete">("idle");

  useEffect(() => {
    if (isSubmitting && phase === "idle") {
      setPhase("loading");
    } else if (!isSubmitting && phase === "loading") {
      setPhase("complete");
      const t = setTimeout(() => {
        setPhase("idle");
        if(onComplete) onComplete();
      }, 1500);
      return () => clearTimeout(t);
    }
  }, [isSubmitting, phase, onComplete]);

  return (
    <div className="w-full flex justify-center text-carbon bg-transparent h-[52px]">
      <AnimatePresence mode="popLayout">
        {phase === "idle" && (
          <motion.div
            key="idle"
            exit={{ width: 48, opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="w-full h-full"
          >
            <button className="w-full h-full rounded-xl bg-gradient-to-br from-grape to-aqua font-sans font-bold text-base flex justify-center items-center">
              Search PlotSense →
            </button>
          </motion.div>
        )}

        {phase === "loading" && (
          <motion.div
            key="loading"
            initial={{ width: "100%", borderRadius: 12 }}
            animate={{ width: 48, borderRadius: 24 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="h-[48px] bg-transparent flex justify-center items-center overflow-hidden relative"
          >
             <motion.div 
               animate={{ rotate: 360 }}
               transition={{ duration: 0.8, ease: "linear", repeat: Infinity }}
               className="w-10 h-10 rounded-full border-2 border-grape border-t-aqua shrink-0 text-transparent"
             />
          </motion.div>
        )}

        {phase === "complete" && (
          <motion.div
            key="complete"
            initial={{ width: 48 }}
            animate={{ width: "100%" }}
            exit={{ opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="h-full bg-aqua/30 rounded-xl flex justify-center items-center text-white font-sans font-bold"
          >
            ✓ Done
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
