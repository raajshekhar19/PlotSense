"use client";

import Nav from "@/components/Nav";
import NodeCanvas from "@/components/graph/NodeCanvas";
import { motion } from "framer-motion";

export default function GraphExplorer() {
  return (
    <main className="relative min-h-screen bg-carbon overflow-hidden selection:bg-aqua selection:text-carbon">
      <Nav />
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-grape/20 via-carbon to-carbon -z-10" />

      <div className="relative z-10 flex flex-col items-center justify-start min-h-[90vh] pt-32 px-4 md:px-12 max-w-[1400px] mx-auto">
        <motion.div
           initial={{ opacity: 0, y: 20 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 0.6 }}
           className="w-full text-center md:text-left mb-12"
        >
          <div className="inline-block px-4 py-1.5 rounded-full bg-grape/30 border border-grape border-opacity-50 text-aqua font-mono text-sm tracking-widest mb-4 backdrop-blur-sm shadow-xl shadow-grape/10">
            NEO4J ENTITY RESOLUTION
          </div>
          <h1 className="font-display text-4xl md:text-6xl text-white tracking-[2px] mb-4">
            Graph Explorer
          </h1>
          <p className="font-sans text-steel text-lg md:text-xl font-light max-w-2xl leading-relaxed">
            PlotSense uses an interconnected Neo4j graph database. When you search for specific actors, directors, or genres, we mathematically traverse millions of relationships to find exact contextual overlaps. Try hovering over the simulated graph below.
          </p>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="w-full relative"
        >
          {/* Main Display Frame */}
          <div className="absolute -inset-[2px] rounded-[26px] bg-gradient-to-br from-aqua/30 via-grape/10 to-transparent blur-md pointer-events-none" />
          
          <div className="w-full rounded-3xl relative backdrop-blur-3xl overflow-hidden border border-white/[0.05] p-2 md:p-6 shadow-2xl bg-carbon/60">
            <NodeCanvas />
            
            {/* Terminal Mock Overlay */}
            <div className="hidden lg:block absolute bottom-12 right-12 w-80 bg-carbon-500/90 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl p-6 pointer-events-none">
              <h4 className="font-mono text-xs text-steel mb-3 tracking-widest uppercase">Live Cypher Query</h4>
              <div className="font-mono text-xs text-aqua leading-relaxed opacity-80">
                <span className="text-white">MATCH</span> (m:Movie)<br/>
                <span className="text-white">WHERE</span> (m)-[:DIRECTED_BY]-&gt;(d:Director)<br/>
                <span className="text-white">AND</span> d.name = <span className="text-grape">"Christopher Nolan"</span><br/>
                <span className="text-white">RETURN</span> m.title, m.plot<br/>
                <span className="text-white">LIMIT</span> 5
              </div>
            </div>
          </div>
        </motion.div>
      </div>
      
      {/* Bottom padding */}
      <div className="h-32 w-full" />
    </main>
  );
}
