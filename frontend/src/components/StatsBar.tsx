"use client";

import { useEffect, useRef } from "react";
import { animate, useInView, useMotionValue, useTransform, motion } from "framer-motion";

function Counter({ from = 0, to, suffix = "", prefix = "" }: { from?: number; to: number; suffix?: string; prefix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const count = useMotionValue(from);
  const rounded = useTransform(count, (latest) => {
    return `${prefix}${Math.round(latest).toLocaleString()}${suffix}`;
  });
  const inView = useInView(ref, { once: true, margin: "-50px" });

  useEffect(() => {
    if (inView) {
      animate(count, to, { duration: 2, ease: "easeOut" });
    }
  }, [count, inView, to]);

  return <motion.span ref={ref}>{rounded}</motion.span>;
}

export default function StatsBar() {
  const containerRef = useRef<HTMLDivElement>(null);
  const inView = useInView(containerRef, { once: true, margin: "-50px" });

  const stats = [
    { num: 45000, suffix: "+", label: "Movies Indexed", prefix: "" },
    { num: 3, label: "AI Models", prefix: "" },
    { custom: "Neo4j", label: "Knowledge Graph" },
    { num: 200, prefix: "<", suffix: "ms", label: "Response" },
  ];

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
      transition={{ delay: 0.2, duration: 0.6 }}
      className="w-full bg-grape/10 border-y border-grape/20 py-8 relative z-20"
    >
      <div className="max-w-7xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8 divide-x divide-grape/20">
        {stats.map((stat, idx) => (
          <div key={idx} className="flex flex-col items-center justify-center pl-6 first:pl-0 text-center">
            <h3 className="font-mono text-2xl md:text-3xl text-aqua font-bold tracking-tight">
              {stat.custom ? (
                <span>{stat.custom}</span>
              ) : (
                <Counter from={0} to={stat.num!} prefix={stat.prefix} suffix={stat.suffix} />
              )}
            </h3>
            <p className="font-mono text-xs md:text-sm text-steel mt-2 uppercase tracking-wide">
              {stat.label}
            </p>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
