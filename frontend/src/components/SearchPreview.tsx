"use client";

import { motion, useMotionValue, useTransform } from "framer-motion";
import { useEffect, useState, useRef } from "react";
import { Search } from "lucide-react";

const QUERIES = [
  {
    text: "a movie where someone grows potatoes on Mars",
    result: { title: "The Martian", year: "2015", match: "99%" },
  },
  {
    text: "psychological thriller with an unreliable narrator from the 90s",
    result: { title: "Fight Club", year: "1999", match: "98%" },
  },
  {
    text: "animated film about loss and memory with a girl and her dog",
    result: { title: "Up", year: "2009", match: "96%" }, // or spirirted away, but Up is a recognizable one
  },
];

export default function SearchPreview() {
  const [queryIndex, setQueryIndex] = useState(0);
  const [typedChars, setTypedChars] = useState(0);
  const [showResult, setShowResult] = useState(false);

  // Mouse parallax tracking
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Map mouse position limits
  const rotateX = useTransform(mouseY, [-500, 500], [4, -4]);
  const rotateY = useTransform(mouseX, [-500, 500], [-4, 4]);
  const translateX = useTransform(mouseX, [-500, 500], [-12, 12]);
  const translateY = useTransform(mouseY, [-500, 500], [-12, 12]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX - window.innerWidth / 2);
      mouseY.set(e.clientY - window.innerHeight / 2);
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [mouseX, mouseY]);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    const currentQuery = QUERIES[queryIndex];

    // Typing effect
    if (typedChars < currentQuery.text.length) {
      timeoutId = setTimeout(() => {
        setTypedChars(typedChars + 1);
      }, 40); // typing speed
    } else {
      // Done typing, wait 0.8s to show result
      if (!showResult) {
        timeoutId = setTimeout(() => {
          setShowResult(true);
        }, 800);
      } else {
        // Wait 3 seconds then go to next query
        timeoutId = setTimeout(() => {
          setShowResult(false);
          setTypedChars(0);
          setQueryIndex((prev) => (prev + 1) % QUERIES.length);
        }, 3000);
      }
    }

    return () => clearTimeout(timeoutId);
  }, [typedChars, showResult, queryIndex]);

  const displayedText = QUERIES[queryIndex].text.substring(0, typedChars);
  const resultData = QUERIES[queryIndex].result;

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1.2, duration: 0.8, ease: "easeOut" }}
      style={{
        rotateX,
        rotateY,
        translateX,
        translateY,
        transformPerspective: 1000,
      }}
      className="relative z-20 mx-auto mt-24 mb-16 w-full max-w-[680px] px-4"
    >
      <div className="bg-grape/20 border border-aqua/30 backdrop-blur-xl rounded-[20px] p-4 shadow-2xl overflow-hidden relative">
        {/* Mock Search Bar */}
        <div className="flex items-center bg-carbon/50 rounded-xl px-4 py-3 border border-steel/20">
          <Search className="w-5 h-5 text-steel mr-3" />
          <span className="font-sans text-white text-base">
            {displayedText}
            <motion.span
              animate={{ opacity: [0, 1, 0] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-[2px] h-5 bg-aqua ml-1 align-middle"
            />
          </span>
        </div>

        {/* Micro-Result Row */}
        <motion.div
          animate={{ opacity: showResult ? 1 : 0, y: showResult ? 0 : 10 }}
          transition={{ duration: 0.4 }}
          className="mt-4 px-2"
        >
          <div className="flex items-center justify-between p-3 rounded-xl bg-grape/30 border border-grape/50 hover:bg-grape/40 transition-colors cursor-pointer">
            <div className="flex items-center gap-4">
              <div className="w-10 h-14 bg-carbon rounded bg-gradient-to-tr from-carbon to-grape border border-steel/20 shadow-inner" />
              <div>
                <h4 className="font-sans font-semibold text-white text-sm">
                  {resultData.title}
                </h4>
                <p className="font-mono text-xs text-steel mt-1">
                  {resultData.year}
                </p>
              </div>
            </div>
            
            <div className="px-3 py-1 bg-carbon/60 rounded-full border border-aqua/20 flex flex-col items-center">
               <span className="font-mono text-[10px] text-steel">MATCH</span>
               <span className="font-mono text-sm text-aqua font-bold">{resultData.match}</span>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
