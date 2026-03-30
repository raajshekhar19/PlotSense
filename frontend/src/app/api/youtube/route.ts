import { NextResponse } from "next/server";

/**
 * Serverless API route to fetch a YouTube video ID for a movie trailer.
 * Uses YouTube's public oEmbed endpoint + InnerTube search as fallback.
 *
 * GET /api/youtube?title=Inception&year=2010
 */
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const title = searchParams.get("title") || "";
  const year = searchParams.get("year") || "";

  if (!title) {
    return NextResponse.json(
      { error: "Missing title parameter" },
      { status: 400 }
    );
  }

  const searchQuery = `${title} ${year} official trailer`.trim();

  try {
    // Approach: scrape YouTube search results page for the first video ID
    const ytSearchUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(
      searchQuery
    )}`;

    const res = await fetch(ytSearchUrl, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
      },
      signal: AbortSignal.timeout(6000),
    });

    if (!res.ok) {
      throw new Error(`YouTube returned ${res.status}`);
    }

    const html = await res.text();

    // Extract videoId from the search results page JSON
    // YouTube embeds initial data as `var ytInitialData = { ... };`
    const videoIdMatch = html.match(/"videoId":"([a-zA-Z0-9_-]{11})"/);

    if (videoIdMatch && videoIdMatch[1]) {
      return NextResponse.json({ videoId: videoIdMatch[1] });
    }

    // Fallback: try regex for /watch?v= links
    const watchMatch = html.match(/\/watch\?v=([a-zA-Z0-9_-]{11})/);
    if (watchMatch && watchMatch[1]) {
      return NextResponse.json({ videoId: watchMatch[1] });
    }

    return NextResponse.json({ error: "No trailer found" }, { status: 404 });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("YouTube API route error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
