"use client";

import Nav from "@/components/Nav";
import TechLayer from "@/components/architecture/TechLayer";
import { motion } from "framer-motion";

export default function Architecture() {
  const stack = [
    {
      title: "Level 1: UI & Experience",
      subtitle: "The Client Layer",
      delay: 0.1,
      items: [
        { name: "Next.js 15", type: "Framework", desc: "React framework delivering fast client-side navigation and optimized App Router architecture." },
        { name: "Tailwind V4", type: "Styling", desc: "Atomic CSS compiler generating the distinct carbon and aqua neon aesthetic globally." },
        { name: "Framer Motion", type: "Animation", desc: "Hardware-accelerated orchestration library powering nodes, timelines, and smooth mount flows." }
      ]
    },
    {
      title: "Level 2: Orchestration",
      subtitle: "Agentic Logic Flow",
      delay: 0.3,
      items: [
        { name: "FastAPI", type: "Backend", desc: "Asynchronous Python web framework securely routing payloads between the browser and LLM memory." },
        { name: "LangGraph", type: "Control Flow", desc: "State-machine framework dictating the multi-agent conversational paths and fallback error handling." },
        { name: "Pydantic", type: "Validation", desc: "Strictly enforcing LLM string extraction into rigidly validated JSON objects for safe rendering." }
      ]
    },
    {
      title: "Level 3: Intelligence",
      subtitle: "Data & Storage",
      delay: 0.5,
      items: [
        { name: "Neo4j", type: "Graph DB", desc: "Knowledge graph mapping millions of dense cinematic entities (Actors, Genres, Directors) logically." },
        { name: "FAISS", type: "Vector Store", desc: "Facebook AI Similarity Search indexing raw movie plot embeddings to allow rapid mathematical querying." },
        { name: "Groq LLaMA 3.3", type: "Inference", desc: "70B parameter versatile model synthesizing insights and parsing query intents live at ~800 tokens/sec." },
        { name: "Tavily", type: "Web Search", desc: "Intelligent fallback scraper utilized instantly if local database vector indexing returns a null set." }
      ]
    }
  ];

  return (
    <main className="relative min-h-screen bg-carbon overflow-hidden selection:bg-aqua selection:text-carbon pb-32">
      <Nav />
      {/* Dynamic Background Noise */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] pointer-events-none mix-blend-overlay" />

      <div className="relative z-10 flex flex-col pt-40 px-6 md:px-12 max-w-7xl mx-auto min-h-screen">
        <motion.div
           initial={{ opacity: 0, y: 30 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 0.8 }}
           className="mb-20 text-left"
        >
          <div className="inline-flex items-center space-x-2 px-4 py-1.5 rounded-full bg-steel/10 border border-steel/20 text-steel font-mono text-sm tracking-widest mb-6">
            <span className="w-2 h-2 rounded-full bg-aqua animate-pulse" />
            <span>SYSTEM ARCHITECTURE</span>
          </div>
          <h1 className="font-display text-5xl md:text-7xl text-white tracking-[2px] leading-tight mb-8">
            How PlotSense <br/> <span className="text-grape">Is Engineered</span>
          </h1>
          <p className="font-sans text-steel text-lg max-w-2xl font-light leading-relaxed">
            PlotSense represents the intersection of state-machine AI Orchestration and Graph Data structures. Explore our multi-layered tech stack below.
          </p>
        </motion.div>

        {/* Vertical Stack Divider (Desktop only) */}
        <div className="relative pl-0 lg:pl-16">
          <div className="hidden lg:block absolute left-[30.5%] top-0 bottom-0 w-[1px] bg-gradient-to-b from-transparent via-grape/40 to-transparent" />
          
          <div className="flex flex-col space-y-24 w-full">
            {stack.map((layer, i) => (
              <TechLayer key={i} {...layer} />
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
