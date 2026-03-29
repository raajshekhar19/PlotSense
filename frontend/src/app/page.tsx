"use client";

import Nav from "@/components/Nav";
import HeroContent from "@/components/HeroContent";
import SearchPreview from "@/components/SearchPreview";
import StatsBar from "@/components/StatsBar";
import HowItWorks from "@/components/HowItWorks";

export default function Home() {
  return (
    <main className="relative min-h-screen bg-carbon overflow-hidden selection:bg-aqua selection:text-carbon">
      {/* Background noise and gradients */}
      <div className="absolute inset-0 z-0 pointer-events-none">
         <div className="absolute inset-0 bg-noise opacity-15" />
         {/* Slow-panning radial gradient emanating from bottom-center in #4B4E6D at 8% opacity */}
         <div 
           className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[120vw] h-[100vh] rounded-full animate-gradient"
           style={{
             background: 'radial-gradient(circle 50vw at 50% 100%, rgba(75, 78, 109, 0.08) 0%, transparent 80%)'
           }}
         />
      </div>

      <Nav />
      
      <div className="relative z-10 flex flex-col items-center justify-center min-h-[90vh]">
        <HeroContent />
        <SearchPreview />
      </div>

      <StatsBar />
      <HowItWorks />
      
      {/* Optional padding at the bottom for smooth scroll ending */}
      <div className="h-24 w-full bg-carbon" />
    </main>
  );
}
