"use client";

import { motion } from "framer-motion";

interface TechNode {
  name: string;
  type: string;
  desc: string;
}

interface TechLayerProps {
  title: string;
  subtitle: string;
  items: TechNode[];
  delay: number;
}

export default function TechLayer({ title, subtitle, items, delay }: TechLayerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -30 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay, ease: "easeOut" }}
      className="relative flex flex-col md:flex-row w-full gap-8 md:gap-12"
    >
      {/* Category Header */}
      <div className="w-full md:w-1/3 flex flex-col pt-4 relative">
        <div className="absolute left-0 top-0 w-8 h-[1px] bg-aqua" />
        <h3 className="font-display text-3xl text-white tracking-widest mt-2">{title}</h3>
        <p className="font-mono text-sm text-steel mt-2 uppercase tracking-widest opacity-80">{subtitle}</p>
      </div>

      {/* Nodes Grid */}
      <div className="w-full md:w-2/3 grid grid-cols-1 sm:grid-cols-2 gap-4">
        {items.map((item, idx) => (
          <motion.div
            key={idx}
            whileHover={{ y: -5, scale: 1.02 }}
            className="group relative bg-[#1A1A1A] rounded-2xl p-6 border border-white/5 hover:border-grape/40 transition-all duration-300 shadow-xl overflow-hidden"
          >
            {/* Shimmer on hover */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-500 bg-gradient-to-tr from-transparent via-aqua to-transparent" />
            
            <div className="flex justify-between items-start mb-4">
              <h4 className="font-sans font-bold text-white text-lg tracking-wide">{item.name}</h4>
              <span className="font-mono text-[10px] text-aqua bg-aqua/10 px-2 py-1 rounded border border-aqua/20 uppercase">
                {item.type}
              </span>
            </div>
            <p className="font-sans font-light text-steel text-sm leading-relaxed">
              {item.desc}
            </p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
