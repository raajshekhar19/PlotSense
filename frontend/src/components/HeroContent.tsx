"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight } from "lucide-react";

export default function HeroContent() {
  const { scrollY } = useScroll();
  const yParallax = useTransform(scrollY, [0, 1000], [0, 300]);

  const h1Lines = ["Find Movies", "By How They", "Feel"];

  return (
    <div className="flex flex-col items-center text-center mt-32 md:mt-48 px-4 z-10 relative">
      {/* Eyebrow Tag */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.6 }}
        className="font-mono text-xs text-aqua bg-grape/40 px-3 py-1.5 rounded"
      >
        SEMANTIC MOVIE SEARCH
      </motion.div>

      {/* H1 Title with Stagger & Parallax */}
      <motion.h1
        style={{ y: yParallax }}
        className="font-display text-white text-6xl md:text-[96px] leading-tight md:leading-[1.0] tracking-[-2px] mt-8 flex flex-col items-center"
      >
        {h1Lines.map((line, idx) => (
          <div key={idx} className="overflow-hidden">
            <motion.span
              initial={{ y: 60, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{
                duration: 0.8,
                ease: [0.16, 1, 0.3, 1],
                delay: 0.4 + idx * 0.12,
              }}
              className="block"
            >
              {line}
            </motion.span>
          </div>
        ))}
      </motion.h1>

      {/* Subheading */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, duration: 0.6 }}
        className="font-sans text-lg text-steel max-w-[560px] mx-auto mt-8 font-normal"
      >
        Describe a plot. Name a vibe. Ask a question. PlotSense uses
        vector embeddings + knowledge graphs to find exactly what you&apos;re thinking of.
      </motion.p>

      {/* CTAs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9, duration: 0.6 }}
        className="flex flex-col sm:flex-row items-center gap-4 mt-12"
      >
        <motion.button
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.97 }}
          className="bg-aqua text-carbon font-sans font-semibold text-base px-8 py-4 rounded-full flex items-center justify-center transition-all hover:brightness-110"
        >
          Start Searching
        </motion.button>
        <motion.button
          whileHover={{ backgroundColor: "rgba(75, 78, 109, 0.2)" }}
          whileTap={{ scale: 0.97 }}
          className="bg-transparent border border-grape text-steel font-sans font-medium text-base px-8 py-4 rounded-full flex items-center justify-center gap-2 transition-colors"
        >
          See Architecture <ArrowRight className="w-4 h-4 ml-1" />
        </motion.button>
      </motion.div>
    </div>
  );
}
