"use client";

import { motion } from "framer-motion";

const steps = [
  {
    num: "01",
    title: "Intent Classification",
    desc: "Every query is intercepted by our Groq LLM router. It instantly decides whether your request is semantic (describing a plot) or structured (looking for specific actors, directors, or genres).",
    color: "from-[#84DCC6] to-[#4B4E6D]",
    delay: 0.1,
  },
  {
    num: "02",
    title: "Intelligent Retrieval",
    desc: "If semantic, we search a high-dimensional FAISS vector database. If structured, we traverse the Neo4j Knowledge Graph looking for exact entity relationships.",
    color: "from-[#4B4E6D] to-[#95A3B3]",
    delay: 0.2,
  },
  {
    num: "03",
    title: "Hybrid Reranking",
    desc: "When both vectors and graph nodes return matches, we use Reciprocal Rank Fusion (RRF) to merge and intelligently rerank the most mathematically relevant results.",
    color: "from-[#95A3B3] to-[#84DCC6]",
    delay: 0.3,
  },
  {
    num: "04",
    title: "Answer Synthesis",
    desc: "The final candidates are passed to an LLM evaluator, which streams back a custom narrative explaining exactly why these movies fit your unique search.",
    color: "from-[#84DCC6] to-[#FFFFFF]",
    delay: 0.4,
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="relative w-full py-32 px-8 z-10">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          className="text-center mb-24"
        >
          <h2 className="font-display text-5xl md:text-7xl text-white tracking-widest mb-6 uppercase">
            The Pipeline
          </h2>
          <p className="font-sans text-steel text-lg md:text-xl max-w-2xl mx-auto font-light">
            How PlotSense interprets natural language and returns precise cinematic intelligence in milliseconds.
          </p>
        </motion.div>

        <div className="relative grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Background connecting line on Desktop */}
          <div className="hidden lg:block absolute top-[60px] left-[10%] right-[10%] h-[1px] bg-gradient-to-r from-aqua/20 via-grape/40 to-white/20 z-0" />

          {steps.map((step, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: step.delay, duration: 0.6, ease: "easeOut" }}
              className="relative z-10 flex flex-col items-center md:items-start lg:items-center text-center md:text-left lg:text-center"
            >
              <div 
                className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${step.color} p-[1px] mb-8 shadow-2xl shadow-carbon-500`}
              >
                <div className="w-full h-full bg-[#1A1A1A] rounded-2xl flex items-center justify-center backdrop-blur-md">
                  <span className="font-mono text-xl font-bold bg-clip-text text-transparent bg-gradient-to-br from-white to-steel">
                    {step.num}
                  </span>
                </div>
              </div>
              
              <h3 className="font-display text-2xl text-white tracking-wider mb-4 uppercase">
                {step.title}
              </h3>
              <p className="font-sans text-steel/80 text-sm leading-relaxed font-light">
                {step.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
