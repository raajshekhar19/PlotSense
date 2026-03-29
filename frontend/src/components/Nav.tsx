"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { label: "Search", href: "/search" },
  { label: "How it Works", href: "/#how-it-works" },
  { label: "Graph Explorer", href: "/graph" },
  { label: "Architecture", href: "/architecture" },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [pulseSearch, setPulseSearch] = useState(false);
  const pathname = usePathname();

  // Scroll listener
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 60);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Keyboard shortcut listener for CTA pulse
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        setPulseSearch(true);
        setTimeout(() => setPulseSearch(false), 300);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <>
      <motion.nav
        initial={{ y: -100 }}
        animate={{ 
          y: 0,
          backgroundColor: scrolled ? "rgba(34, 34, 34, 0.9)" : "transparent",
          borderBottom: scrolled ? "1px solid rgba(75, 78, 109, 0.4)" : "1px solid transparent",
          height: scrolled ? 60 : 72,
          boxShadow: scrolled ? "0 4px 32px rgba(0,0,0,0.3)" : "none"
        }}
        transition={{ duration: 0.25, ease: "easeInOut" }}
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 backdrop-blur-xl"
      >
        {/* LOGO */}
        <Link href="/" className="flex items-center group relative z-50">
          <span className="font-display text-[28px] tracking-[1px] text-white">PlotSense</span>
          <motion.span
            className="w-[6px] h-[6px] rounded-full bg-aqua ml-1 relative"
            variants={{ hover: { scale: 1.8 } }}
            whileHover="hover"
          >
            <motion.span
              variants={{
                hover: { scale: 1.5, opacity: 0, transition: { duration: 0.6 } }
              }}
              className="absolute inset-0 rounded-full bg-aqua opacity-50 block origin-center"
            />
          </motion.span>
        </Link>

        {/* DESKTOP CENTER LINKS */}
        <div className="hidden md:flex items-center gap-8 relative z-50">
          {LINKS.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.label}
                href={link.href}
                className={`relative font-sans text-sm transition-colors duration-150 py-2 ${
                  isActive ? "text-white" : "text-steel hover:text-white"
                }`}
              >
                {link.label}
                {isActive && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute -bottom-1 left-0 right-0 h-[2px] bg-aqua rounded-full"
                    transition={{ type: "spring", stiffness: 350, damping: 30 }}
                  />
                )}
              </Link>
            );
          })}
        </div>

        {/* DESKTOP RIGHT ACTIONS */}
        <div className="hidden md:flex items-center z-50">
          <a
            href="https://github.com/raajshekhar19/PlotSense"
            target="_blank"
            rel="noopener noreferrer"
            className="font-sans text-sm text-steel hover:text-aqua transition-colors"
          >
            GitHub ↗
          </a>

          {/* Separator */}
          <div className="w-[1px] h-[20px] bg-grape/40 mx-4" />

          {/* CTA Search Button */}
          <Link href="/search">
            <motion.button
              whileHover={{ scale: 1.02, backgroundColor: "#FFFFFF" }}
              whileTap={{ scale: 0.97 }}
              animate={pulseSearch ? { boxShadow: "0 0 0 8px rgba(132, 220, 198, 0.4)" } : {}}
              className="bg-aqua text-carbon font-sans font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              Start Searching
            </motion.button>
          </Link>

          {/* API STATUS DOT */}
          <div className="relative ml-6 group flex items-center justify-center">
            <motion.div
              animate={{ scale: [1, 1.4, 1], opacity: [1, 0, 1] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              className="absolute w-2 h-2 rounded-full bg-aqua"
            />
            <div className="relative w-2 h-2 bg-aqua rounded-full z-10" />

            {/* Tooltip */}
            <div className="absolute top-8 right-0 -translate-x-1/4 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap bg-carbon border border-grape rounded-md px-3 py-[6px] shadow-xl z-50">
              <span className="font-sans text-xs text-steel">API Status: <strong className="text-aqua font-normal">Online · 142ms</strong></span>
            </div>
          </div>
        </div>

        {/* MOBILE MENU TOGGLE */}
        <button
          className="md:hidden relative z-50 text-steel p-2 -mr-2"
          onClick={() => setIsOpen(!isOpen)}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={isOpen ? "close" : "open"}
              initial={{ opacity: 0, rotate: -90 }}
              animate={{ opacity: 1, rotate: 0 }}
              exit={{ opacity: 0, rotate: 90 }}
              transition={{ duration: 0.2 }}
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </motion.div>
          </AnimatePresence>
        </button>
      </motion.nav>

      {/* MOBILE DRAWER FULL SCREEN */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: "0%" }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 350, damping: 40 }}
            className="fixed inset-0 bg-carbon z-40 flex flex-col justify-center px-8"
          >
            <div className="flex flex-col gap-10 mt-16">
              {LINKS.map((link) => (
                <Link
                  key={link.label}
                  href={link.href}
                  onClick={() => setIsOpen(false)}
                  className="font-sans text-[20px] text-white tracking-tight active:text-aqua transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </div>

            <div className="mt-auto mb-16 flex flex-col gap-6">
              <a
                href="https://github.com/raajshekhar19/PlotSense"
                target="_blank"
                rel="noopener noreferrer"
                className="font-sans text-[20px] text-steel"
              >
                GitHub ↗
              </a>
              <Link href="/search" onClick={() => setIsOpen(false)}>
                <button className="bg-aqua text-carbon w-full font-sans font-semibold text-[16px] py-4 rounded-xl">
                  Start Searching
                </button>
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
