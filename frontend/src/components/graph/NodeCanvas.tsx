"use client";

import { motion } from "framer-motion";
import { useState } from "react";

// Simulated Graph Data representing the Neo4j ontology
const nodes = [
  { id: "m1", label: "Inception", type: "Movie", x: 50, y: 50, color: "from-[#84DCC6] to-[#2B9A80]" },
  { id: "d1", label: "Christopher Nolan", type: "Director", x: 25, y: 20, color: "from-[#4B4E6D] to-[#2C2E43]" },
  { id: "a1", label: "Leonardo DiCaprio", type: "Actor", x: 75, y: 25, color: "from-[#95A3B3] to-[#5F6A76]" },
  { id: "g1", label: "Sci-Fi", type: "Genre", x: 30, y: 80, color: "from-[#FADB5F] to-[#C7A92C]" },
  { id: "g2", label: "Thriller", type: "Genre", x: 70, y: 80, color: "from-[#FADB5F] to-[#C7A92C]" },
  { id: "r1", label: "Interstellar", type: "Movie", x: 15, y: 55, color: "from-[#84DCC6] to-[#2B9A80]" },
  { id: "r2", label: "The Revenant", type: "Movie", x: 85, y: 50, color: "from-[#84DCC6] to-[#2B9A80]" },
];

const edges = [
  { source: "d1", target: "m1", label: ":DIRECTED_BY" },
  { source: "d1", target: "r1", label: ":DIRECTED_BY" },
  { source: "a1", target: "m1", label: ":ACTED_IN" },
  { source: "a1", target: "r2", label: ":ACTED_IN" },
  { source: "m1", target: "g1", label: ":BELONGS_TO" },
  { source: "m1", target: "g2", label: ":BELONGS_TO" },
];

export default function NodeCanvas() {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Helper to find edge transparency based on hover state
  const getEdgeOpacity = (sourceId: string, targetId: string) => {
    if (!hoveredNode) return 0.2;
    if (sourceId === hoveredNode || targetId === hoveredNode) return 0.8;
    return 0.05;
  };

  return (
    <div className="relative w-full h-[600px] lg:h-[800px] bg-carbon rounded-3xl border border-grape/30 overflow-hidden shadow-2xl flex items-center justify-center">
      {/* Background Grid */}
      <div 
        className="absolute inset-0 opacity-[0.03] pointer-events-none" 
        style={{ backgroundImage: 'radial-gradient(#84DCC6 1px, transparent 1px)', backgroundSize: '40px 40px' }} 
      />

      {/* SVG Canvas for Edges */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        
        {edges.map((edge, idx) => {
          const source = nodes.find(n => n.id === edge.source)!;
          const target = nodes.find(n => n.id === edge.target)!;
          const isHighlighted = hoveredNode === source.id || hoveredNode === target.id;
          
          return (
            <g key={idx}>
              <motion.line
                x1={`${source.x}%`}
                y1={`${source.y}%`}
                x2={`${target.x}%`}
                y2={`${target.y}%`}
                stroke={isHighlighted ? "#84DCC6" : "#4B4E6D"}
                strokeWidth={isHighlighted ? 2 : 1}
                animate={{ opacity: getEdgeOpacity(edge.source, edge.target) }}
                transition={{ duration: 0.3 }}
                filter={isHighlighted ? "url(#glow)" : ""}
              />
              {/* Relationship Label (only visible on relationship hover) */}
              <motion.text
                x={`${(source.x + target.x) / 2}%`}
                y={`${(source.y + target.y) / 2 - 2}%`}
                fill={isHighlighted ? "#84DCC6" : "#4B4E6D"}
                fontSize="12"
                fontFamily="JetBrains Mono"
                textAnchor="middle"
                animate={{ opacity: isHighlighted ? 1 : 0 }}
              >
                {edge.label}
              </motion.text>
            </g>
          );
        })}
      </svg>

      {/* HTML Canvas for Nodes */}
      {nodes.map((node, idx) => {
        const isHovered = hoveredNode === node.id;
        const isDimmed = hoveredNode && hoveredNode !== node.id && !edges.find(e => (e.source === node.id && e.target === hoveredNode) || (e.target === node.id && e.source === hoveredNode));
        
        return (
          <motion.div
            key={node.id}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ 
              scale: isHovered ? 1.2 : 1, 
              opacity: isDimmed ? 0.3 : 1,
              y: ["-3%", "3%", "-3%"] // Floating effect
            }}
            transition={{ 
              scale: { duration: 0.2 }, 
              opacity: { duration: 0.3 },
              y: { duration: 4 + (idx % 3), repeat: Infinity, ease: "easeInOut" }
            }}
            className="absolute z-10 flex flex-col items-center cursor-crosshair group"
            style={{ left: `${node.x}%`, top: `${node.y}%`, transform: 'translate(-50%, -50%)' }}
            onMouseEnter={() => setHoveredNode(node.id)}
            onMouseLeave={() => setHoveredNode(null)}
          >
            {/* The Node Circle */}
            <div className={`relative w-16 h-16 md:w-20 md:h-20 rounded-full bg-gradient-to-br ${node.color} shadow-lg shadow-black/50 flex items-center justify-center p-1 hover:shadow-aqua/20 hover:shadow-2xl transition-all duration-300`}>
              <div className="w-full h-full bg-[#1A1A1A] rounded-full flex items-center justify-center">
                <span className="font-mono text-xs text-white/80 uppercase tracking-widest">{node.type}</span>
              </div>
            </div>
            
            {/* Node Label Tooltip */}
            <motion.div 
              className="absolute top-full mt-4 bg-carbon-500/90 backdrop-blur-md px-4 py-2 rounded-xl border border-white/10 whitespace-nowrap shadow-xl pointer-events-none"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: isHovered ? 1 : 0.8, y: 0 }}
            >
              <h4 className="font-sans text-white text-sm font-semibold">{node.label}</h4>
            </motion.div>
          </motion.div>
        );
      })}
    </div>
  );
}
