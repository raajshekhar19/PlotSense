"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Clipboard, Check } from "lucide-react";

interface Props {
  answerText: string;
}

export default function AnswerSynthesis({ answerText }: Props) {
  const [copied, setCopied] = useState(false);
  const [visibleText, setVisibleText] = useState("");
  const [isStreaming, setIsStreaming] = useState(true);

  // Stream the real answer text character by character
  useEffect(() => {
    setVisibleText("");
    setIsStreaming(true);
    let i = 0;
    const interval = setInterval(() => {
      setVisibleText(answerText.substring(0, i));
      i++;
      if (i > answerText.length) {
        clearInterval(interval);
        setIsStreaming(false);
      }
    }, 8); // faster streaming for real answers which can be long
    return () => clearInterval(interval);
  }, [answerText]);

  const handleCopy = () => {
    navigator.clipboard.writeText(answerText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full relative flex flex-col items-center">
      <AnimatePresence>
        <motion.div
           initial={{ opacity: 0, y: -16, height: 0 }}
           animate={{ opacity: 1, y: 0, height: "auto" }}
           transition={{ type: "spring", stiffness: 300, damping: 30 }}
           className="w-full bg-grape/10 border border-aqua/25 rounded-[16px] p-5 md:p-6 relative overflow-hidden"
        >
          {/* Top Bar */}
          <div className="flex justify-between items-center mb-4">
             <div className="flex items-center gap-2 bg-aqua/15 border border-aqua/40 px-3 py-1 rounded-full text-aqua font-mono text-[11px] uppercase tracking-wide">
               <motion.span 
                 animate={{ rotate: 360 }} 
                 transition={{ duration: 4, ease: "linear", repeat: Infinity }}
                 className="origin-center block"
               >
                 ✦
               </motion.span>
               PlotSense AI
             </div>

             <button 
               onClick={handleCopy}
               className="flex items-center gap-1.5 text-[12px] text-steel hover:text-aqua transition-colors"
             >
               {copied ? <Check className="w-3.5 h-3.5" /> : <Clipboard className="w-3.5 h-3.5" />}
               {copied ? "Copied!" : "Copy answer"}
             </button>
          </div>

          {/* Answer Content */}
          <div className="font-sans text-[16px] text-white/90 leading-[1.75] mb-4 min-h-[48px]">
             {visibleText}
             {isStreaming && (
                <motion.span 
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                  className="inline-block w-[2px] h-4 bg-aqua ml-1 align-baseline rounded-[1px]"
                />
             )}
          </div>

          {/* Sources Footer */}
          <div className="pt-3 mt-3 border-t border-grape/40 flex flex-wrap justify-between items-center gap-4">
             <div className="flex items-center gap-3">
                <span className="font-sans text-[12px] text-steel">Sources used:</span>
                <div className="flex gap-2 font-mono text-[11px]">
                   <span className="bg-grape/30 border border-grape px-2 py-0.5 rounded text-steel flex items-center gap-1">
                     <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                     FAISS Vector DB
                   </span>
                   <span className="bg-aqua/10 border border-aqua/30 px-2 py-0.5 rounded text-aqua flex items-center gap-1">
                     <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>
                     Neo4j KG
                   </span>
                </div>
             </div>

             {/* Confidence Meter */}
             <div className="flex items-center gap-3">
                <span className="font-sans text-[12px] text-steel">Confidence</span>
                <div className="w-[120px] h-[4px] bg-carbon/50 rounded-full overflow-hidden">
                   <motion.div 
                     initial={{ width: 0 }}
                     animate={{ width: "94%" }}
                     transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
                     className="h-full bg-aqua"
                   />
                </div>
                <span className="font-mono text-[11px] text-white">94%</span>
             </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Follow-up Suggestions outside */}
      <div className="w-full flex gap-3 mt-4 flex-wrap pb-4">
         {["What are similar movies by the same director?", "Show me the full cast details"].map((q, idx) => (
           <motion.button
             key={q}
             initial={{ opacity: 0, y: 10 }}
             animate={!isStreaming ? { opacity: 1, y: 0 } : { opacity: 0, y: 10 }}
             transition={{ delay: idx * 0.12 }}
             className="bg-transparent border border-grape/60 text-steel font-sans text-[13px] px-4 py-2 rounded-full hover:bg-grape/20 hover:border-aqua/60 hover:text-white transition-colors"
           >
             {q}
           </motion.button>
         ))}
      </div>
    </div>
  );
}
