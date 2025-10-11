"use client";

import Image from "next/image";
import { useEffect, useState, FormEvent } from "react";

export default function Home() {
  const [studentId, setStudentId] = useState("");
  const [level, setLevel] = useState("beginner");
  const [context, setContext] = useState("SQL basics");
  const [goal, setGoal] = useState("Understand GROUP BY");
  const [useLLM, setUseLLM] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<unknown | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useRealApi, setUseRealApi] = useState<boolean | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
      // Decide endpoint: real or demo. If useRealApi is null (unknown), default to demo.
      const preferReal = Boolean(useRealApi);
      const endpoint = preferReal ? "/api/generate_path" : "/api/generate_path_demo";

      const res = await fetch(`${base}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_id: studentId || "demo-student",
          level,
          context,
          student_goal: goal,
          use_llm: useLLM,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error ${res.status}: ${text}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err: unknown) {
      // Normalize unknown error to string
      if (err instanceof Error) setError(err.message);
      else setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  function safeStringify(v: unknown) {
    try {
      return JSON.stringify(v, null, 2);
    } catch {
      return String(v);
    }
  }

  // On mount, query backend /api/status to detect whether real pipeline can be used.
  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

    // Allow manual override persisted in localStorage
    const stored = typeof window !== "undefined" ? window.localStorage.getItem("plp_use_real_api") : null;
    if (stored === "true") {
      setUseRealApi(true);
      return;
    }
    if (stored === "false") {
      setUseRealApi(false);
      return;
    }

    // Otherwise query backend for status
    fetch(`${base}/api/status`).then(async (res) => {
      try {
        if (!res.ok) {
          setUseRealApi(false);
          return;
        }
        const j = await res.json();
        setUseRealApi(Boolean(j.real_enabled));
      } catch {
        setUseRealApi(false);
      }
    }).catch(() => setUseRealApi(false));
  }, []);

  function toggleUseRealApi(value: boolean) {
    setUseRealApi(value);
    try {
      window.localStorage.setItem("plp_use_real_api", value ? "true" : "false");
    } catch {}
  }

  return (
    <div className="font-sans min-h-screen p-8 sm:p-20">
      <main className="max-w-2xl mx-auto">
        <div className="flex items-center gap-4 mb-6">
          <Image src="/next.svg" alt="Next" width={120} height={26} />
          <h1 className="text-2xl font-semibold">Personalized Learning Path — Demo</h1>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium">Student ID</label>
            <input
              className="mt-1 block w-full border rounded px-3 py-2"
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              placeholder="student-123"
            />
          </div>

          <div>
            <label className="block text-sm font-medium">Level</label>
            <select
              value={level}
              onChange={(e) => setLevel(e.target.value)}
              className="mt-1 block w-full border rounded px-3 py-2"
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium">Context</label>
            <input
              className="mt-1 block w-full border rounded px-3 py-2"
              value={context}
              onChange={(e) => setContext(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-medium">Student Goal</label>
            <input
              className="mt-1 block w-full border rounded px-3 py-2"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
            />
          </div>

          <div className="flex items-center gap-2">
            <input
              id="useLLM"
              type="checkbox"
              checked={useLLM}
              onChange={(e) => setUseLLM(e.target.checked)}
            />
            <label htmlFor="useLLM" className="text-sm">
              Use LLM (may be disabled server-side)
            </label>
          </div>

          <div className="flex items-center gap-2">
            <input
              id="useRealApi"
              type="checkbox"
              checked={Boolean(useRealApi)}
              onChange={(e) => toggleUseRealApi(e.target.checked)}
            />
            <label htmlFor="useRealApi" className="text-sm">
              Use real API when available
            </label>
            <span className="text-xs text-gray-500 ml-2">
              {useRealApi === null ? "(detecting...)" : useRealApi ? "(real)" : "(demo)"}
            </span>
          </div>

          <div>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded"
              disabled={loading}
            >
              {loading ? "Generating…" : "Generate Path"}
            </button>
          </div>
        </form>

        <section className="mt-8">
          <h2 className="text-lg font-medium">Result</h2>
          {error && <pre className="text-red-600 mt-2">{error}</pre>}
          {result !== null && (
            <pre className="mt-2 bg-gray-50 p-4 rounded overflow-auto">{safeStringify(result)}</pre>
          )}
          {!result && !error && <p className="text-sm text-muted mt-2">No result yet.</p>}
        </section>
      </main>
    </div>
  );
}
