"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import styles from "./MovieDetailModal.module.css";

/* ════════════════════════════════════════════
   Types
   ════════════════════════════════════════════ */

interface MatchResult {
  id: string;
  title: string;
  year: string;
  genre: string;
  snippet: string;
  score: number;
  sources: string[];
  director: string;
  posterUrl?: string;
  reason?: string;
}

interface CardRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

interface Props {
  result: MatchResult;
  cardRect: CardRect | null;
  onClose: () => void;
}

/* ════════════════════════════════════════════
   Helper: snippet → render highlighted spans
   ════════════════════════════════════════════ */

function renderSnippet(text: string) {
  return text.split(/(<mark>.*?<\/mark>)/g).map((part, i) => {
    if (part.startsWith("<mark>") && part.endsWith("</mark>")) {
      return (
        <span key={i} className="bg-aqua/20 text-aqua rounded-[3px] px-[2px]">
          {part.replace(/<\/?mark>/g, "")}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

/* ════════════════════════════════════════════
   MovieDetailModal Component
   ════════════════════════════════════════════ */

export default function MovieDetailModal({ result: r, cardRect, onClose }: Props) {
  const [trailerState, setTrailerState] = useState<"idle" | "loading" | "playing" | "error">("idle");
  const [videoId, setVideoId] = useState<string | null>(null);

  // ── Escape key handler ──────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handler);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  // ── Fetch YouTube trailer → open in new tab ──
  const fetchTrailer = useCallback(async () => {
    setTrailerState("loading");
    try {
      const res = await fetch(
        `/api/youtube?title=${encodeURIComponent(r.title)}&year=${encodeURIComponent(r.year)}`
      );
      const data = await res.json();

      if (data.videoId) {
        setVideoId(data.videoId);
        setTrailerState("idle");
        // Open YouTube in a new tab
        window.open(`https://www.youtube.com/watch?v=${data.videoId}`, "_blank", "noopener,noreferrer");
      } else {
        setTrailerState("error");
      }
    } catch {
      setTrailerState("error");
    }
  }, [r.title, r.year]);

  // ── Compute shared-element animation origin ──
  const hasRect = cardRect && cardRect.width > 0;
  const vpW = typeof window !== "undefined" ? window.innerWidth : 1200;
  const vpH = typeof window !== "undefined" ? window.innerHeight : 800;

  const initialX = hasRect ? cardRect!.left + cardRect!.width / 2 - vpW / 2 : 0;
  const initialY = hasRect ? cardRect!.top + cardRect!.height / 2 - vpH / 2 : 40;
  const initialScaleX = hasRect ? cardRect!.width / Math.min(896, vpW - 32) : 0.92;
  const initialScaleY = hasRect ? cardRect!.height / (vpH * 0.85) : 0.92;

  return (
    <AnimatePresence>
      {/* ── Backdrop ────────────────────── */}
      <motion.div
        className={styles.backdrop}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.25 }}
        onClick={onClose}
      >
        {/* ── Modal ─────────────────────── */}
        <motion.div
          className={styles.modal}
          onClick={(e) => e.stopPropagation()}
          initial={{
            opacity: 0,
            x: initialX,
            y: initialY,
            scaleX: initialScaleX,
            scaleY: initialScaleY,
          }}
          animate={{
            opacity: 1,
            x: 0,
            y: 0,
            scaleX: 1,
            scaleY: 1,
          }}
          exit={{
            opacity: 0,
            x: initialX,
            y: initialY,
            scaleX: initialScaleX,
            scaleY: initialScaleY,
          }}
          transition={{
            type: "spring",
            stiffness: 320,
            damping: 34,
            mass: 0.8,
          }}
        >
          {/* ── Blurred Poster Background ── */}
          {r.posterUrl && (
            <div
              className={styles.posterBg}
              style={{ backgroundImage: `url(${r.posterUrl})` }}
            />
          )}
          <div className={styles.posterOverlay} />

          {/* ── Close Button ────────────── */}
          <button className={styles.closeBtn} onClick={onClose}>
            ✕
          </button>

          {/* ── Scrollable Content Area ─── */}
          <div className={styles.content}>
            {/* ── Top Section: Poster + Info ─── */}
            <div className="flex gap-6 md:gap-8 flex-col md:flex-row">
              {/* Sharp poster / trailer embed */}
              <div className="relative self-center md:self-start">
                <div className={styles.posterSharp}>
                  {r.posterUrl ? (
                    <img
                      src={r.posterUrl}
                      alt={r.title}
                      className={styles.posterSharpImg}
                    />
                  ) : (
                    <div className={styles.posterPlaceholder}>
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="rgba(132,220,198,0.35)" strokeWidth="2">
                        <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
                        <line x1="7" y1="2" x2="7" y2="22" />
                        <line x1="17" y1="2" x2="17" y2="22" />
                        <line x1="2" y1="12" x2="22" y2="12" />
                        <line x1="2" y1="7" x2="7" y2="7" />
                        <line x1="2" y1="17" x2="7" y2="17" />
                        <line x1="17" y1="17" x2="22" y2="17" />
                        <line x1="17" y1="7" x2="22" y2="7" />
                      </svg>
                    </div>
                  )}

                  {/* Loading spinner while fetching trailer */}
                  {trailerState === "loading" && (
                    <div className={styles.spinnerOverlay}>
                      <div className={styles.spinner} />
                    </div>
                  )}
                </div>
              </div>

              {/* Movie info */}
              <div className="flex-1 flex flex-col justify-center min-w-0">
                <h2 className="text-2xl md:text-3xl font-bold text-white mb-2 font-sans leading-tight">
                  {r.title}
                </h2>

                <div className="font-mono text-steel flex flex-wrap gap-2 text-sm mb-3">
                  {r.year && <span>{r.year}</span>}
                  {r.year && r.genre && <span className="text-grape">•</span>}
                  {r.genre && <span>{r.genre}</span>}
                  {r.director && (
                    <>
                      <span className="text-grape">•</span>
                      <span className="text-steel/70">Dir. {r.director}</span>
                    </>
                  )}
                </div>

                {/* Meta pills */}
                <div className={styles.metaPills}>
                  {r.sources.map((s) => (
                    <div
                      key={s}
                      className={`${styles.pill} ${
                        s === "FAISS" ? styles.pillFaiss : styles.pillAqua
                      }`}
                    >
                      {s}
                    </div>
                  ))}
                  <div className={`${styles.pill} ${styles.pillScore}`}>
                    Match Score:{" "}
                    <strong className="text-aqua ml-1">{r.score}%</strong>
                  </div>
                </div>

                {/* Trailer button */}
                <div className="mt-2 flex flex-wrap items-center gap-3">
                  <button
                    className={styles.trailerBtn}
                    onClick={fetchTrailer}
                    disabled={trailerState === "loading"}
                  >
                    {trailerState === "loading" ? (
                      <>
                        <div className={styles.spinner} style={{ width: 14, height: 14, borderWidth: 2 }} />
                        Loading…
                      </>
                    ) : (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                          <polygon points="5,3 19,12 5,21" />
                        </svg>
                        Watch Trailer
                      </>
                    )}
                  </button>

                  {trailerState === "error" && (
                    <span className={styles.trailerError}>Trailer not found</span>
                  )}
                </div>
              </div>
            </div>

            {/* ── Plot Summary ──────────────── */}
            <div className="mt-8 border-t border-white/[0.08] pt-6">
              <h3 className="text-xs font-bold text-white/80 mb-3 uppercase tracking-[0.12em] font-mono">
                Plot Summary
              </h3>
              <p className="text-[15px] text-steel/90 leading-relaxed font-sans whitespace-pre-wrap">
                {renderSnippet(r.snippet).map((part, i) => (
                  <span key={i}>{part}</span>
                ))}
              </p>
            </div>

            {/* ── "Why This Result?" Section ── */}
            {r.reason && r.reason.trim().length > 0 && (
              <div className={styles.reasonSection}>
                <div className={styles.reasonLabel}>AI Insight</div>
                <div className={styles.reasonBlock}>
                  <div className={styles.reasonBar} />
                  <p className={styles.reasonText}>{r.reason}</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
