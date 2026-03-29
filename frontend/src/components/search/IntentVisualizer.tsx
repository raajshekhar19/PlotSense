"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { Check } from "lucide-react";

const NODES = [
  { id: 0, label: "Tokenizing" },
  { id: 1, label: "Extracting Entities" },
  { id: 2, label: "Classifying Intent" },
  { id: 3, label: "Routing Query" },
  { id: 4, label: "Fetching Results" },
];

interface Props {
  isAnalyzing: boolean;
  hasSubmitted: boolean;
}

export default function IntentVisualizer({ isAnalyzing, hasSubmitted }: Props) {
  const [activeStep, setActiveStep] = useState(-1);

  useEffect(() => {
    let timeouts: NodeJS.Timeout[] = [];
    if (isAnalyzing && !hasSubmitted) {
      setActiveStep(0);
      timeouts.push(setTimeout(() => setActiveStep(1), 300));
      timeouts.push(setTimeout(() => setActiveStep(2), 600));
      timeouts.push(setTimeout(() => setActiveStep(3), 900));
    } else if (!isAnalyzing && !hasSubmitted) {
      setActiveStep(-1);
    } else if (hasSubmitted) {
      setActiveStep(4);
    }

    return () => timeouts.forEach(clearTimeout);
  }, [isAnalyzing, hasSubmitted]);

  return (
    <AnimatePresence>
      {(isAnalyzing || hasSubmitted) && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="w-full relative mt-4 pb-12 overflow-x-auto no-scrollbar"
        >
          <div className="flex items-center min-w-[640px] h-[80px] px-2 relative">
            {NODES.map((node, i) => {
              const isActive = activeStep === i && !hasSubmitted;
              const isCompleted = activeStep > i || (hasSubmitted && i < 4);
              const isFinalFetching = i === 4 && hasSubmitted;

              return (
                <div key={node.id} className="flex items-center relative z-10">
                  {/* Node */}
                  <div
                    className={`relative w-[110px] h-[44px] flex items-center justify-center rounded-lg transition-all duration-300 font-sans text-xs shrink-0
                      ${
                        isFinalFetching || isCompleted
                           ? "bg-aqua/15 border border-aqua/50"
                           : isActive
                           ? "bg-grape/50 border-[1.5px] border-aqua shadow-[0_0_16px_rgba(132,220,198,0.3)] text-white"
                           : "bg-grape/20 border border-grape/50 text-steel"
                      }
                    `}
                  >
                     {isCompleted || isFinalFetching ? (
                       <div className="flex items-center text-aqua font-medium w-full text-center justify-center">
                          {isFinalFetching ? <span className="mr-1 animate-pulse">●</span> : null}
                          {node.label}
                          {(isCompleted && !isFinalFetching) && <Check className="w-3 h-3 ml-1" />}
                       </div>
                     ) : (
                       <span>{node.label}</span>
                     )}
                  </div>

                  {/* Connecting Line */}
                  {i < NODES.length - 1 && (
                    <div className="w-[30px] h-0 relative -z-10 shrink-0 mx-[2px]">
                      <svg className="absolute w-full top-1/2 -translate-y-1/2 overflow-visible">
                        <line
                          x1="0"
                          y1="0"
                          x2="100%"
                          y2="0"
                          stroke={isCompleted ? "#84DCC6" : "rgba(75,78,109,0.3)"}
                          strokeWidth="2"
                          strokeDasharray="4 4"
                          className={isCompleted ? "animate-line-flow" : ""}
                        />
                      </svg>
                    </div>
                  )}

                  {/* Extras per node */}
                  
                  {/* Extracted Entities */}
                  {i === 1 && (activeStep >= 1) && (
                    <div className="absolute top-[52px] left-0 flex gap-2 w-[200px]">
                      {["Comedy", "1990s"].map((ent, eIdx) => (
                         <motion.div
                           key={ent}
                           initial={{ scale: 0.5, opacity: 0 }}
                           animate={{ scale: 1, opacity: 1 }}
                           transition={{ delay: eIdx * 0.07 }}
                           className="bg-aqua/10 border border-aqua/40 text-aqua text-[10px] font-mono px-2 py-0.5 rounded shrink-0 whitespace-nowrap"
                         >
                           {ent}
                         </motion.div>
                      ))}
                    </div>
                  )}

                  {/* Intent Badge */}
                  {i === 2 && (activeStep >= 2) && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 500, damping: 20 }}
                      className="absolute top-[52px] left-0 bg-grape border border-aqua/50 text-aqua font-mono text-[10px] px-2 py-1 rounded-full whitespace-nowrap"
                    >
                      ✨ Hybrid Query
                    </motion.div>
                  )}

                  {/* Routing Strategy */}
                  {i === 3 && (activeStep >= 3) && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="absolute top-[54px] left-0 text-steel text-[10px] whitespace-nowrap bg-carbon/80 px-1 rounded"
                    >
                      Strategy: FAISS semantic + Neo4j traversal
                    </motion.div>
                  )}
                </div>
              );
            })}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
