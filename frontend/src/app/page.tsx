"use client";

// logo removed on request
import { useEffect, useState, FormEvent, useRef } from "react";

type ChatMessage = {
  role: string;
  content: string;
  timestamp?: string;
};

type PathNode = {
  id?: string;
  node_id?: string;
  title?: string;
  concept?: string;
  estimated_minutes?: number;
  time_estimate?: number;
};

type SupportingNode = {
  node_id?: string;
  concept?: string;
  title?: string;
  objective?: string;
  context?: string;
  mastery?: number;
  deficiency?: number;
};

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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatInsights, setChatInsights] = useState<SupportingNode[]>([]);
  const [chatSummary, setChatSummary] = useState<string>("");
  const [chatModel, setChatModel] = useState<string>("heuristic");
  const [chatFallback, setChatFallback] = useState<boolean>(false);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages]);

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

  async function handleChatSubmit(e: FormEvent) {
    e.preventDefault();
    if (!chatInput.trim()) {
      return;
    }
    setChatLoading(true);
    setChatError(null);

    try {
      const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
      const res = await fetch(`${base}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: chatInput,
          session_id: chatSessionId,
          learner_id: studentId || undefined,
          context,
          goal,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error ${res.status}: ${text}`);
      }

      const data = await res.json();
      if (data.status !== "ok") {
        throw new Error(data.message || "Chat request failed");
      }

      setChatSessionId(data.session_id || null);
      const history = Array.isArray(data.chat_history)
        ? data.chat_history.map((item: Record<string, unknown>) => ({
            role: String(item.role ?? "assistant"),
            content: String(item.content ?? ""),
            timestamp:
              typeof item.timestamp === "string" ? item.timestamp : undefined,
          }))
        : [];
      setChatMessages(history);

      const insights = Array.isArray(data.supporting_nodes)
        ? data.supporting_nodes
            .map((node: Record<string, unknown>) => ({
              node_id: typeof node.node_id === "string" ? node.node_id : undefined,
              concept: typeof node.concept === "string" ? node.concept : undefined,
              title: typeof node.title === "string" ? node.title : undefined,
              objective: typeof node.objective === "string" ? node.objective : undefined,
              context: typeof node.context === "string" ? node.context : undefined,
              mastery: typeof node.mastery === "number" ? node.mastery : undefined,
              deficiency:
                typeof node.deficiency === "number" ? node.deficiency : undefined,
            }))
            .filter((node: SupportingNode) => Boolean(node.concept || node.title))
        : [];
      setChatInsights(insights);
      setChatSummary(data.summary || "");
      setChatModel(data.model || "heuristic");
      setChatFallback(Boolean(data.fallback));
      setChatInput("");
    } catch (err: unknown) {
      if (err instanceof Error) setChatError(err.message);
      else setChatError(String(err));
    } finally {
      setChatLoading(false);
    }
  }

  function handleChatReset() {
    setChatMessages([]);
    setChatSessionId(null);
    setChatInsights([]);
    setChatSummary("");
    setChatModel("heuristic");
    setChatFallback(false);
    setChatError(null);
    setChatInput("");
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

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages]);

  function toggleUseRealApi(value: boolean) {
    setUseRealApi(value);
    try {
      window.localStorage.setItem("plp_use_real_api", value ? "true" : "false");
    } catch {}
  }

  const pathNodes: PathNode[] | null = (() => {
    if (!result || typeof result !== "object") {
      return null;
    }
    const maybe = (result as { path?: unknown }).path;
    if (!Array.isArray(maybe)) {
      return null;
    }
    return maybe as PathNode[];
  })();
  const hasPath = Boolean(pathNodes && pathNodes.length > 0);

  return (
    <div className="font-sans min-h-screen p-8 sm:p-20">
      <div className={`flex flex-col lg:flex-row ${hasPath ? "gap-8" : ""}`}>
        {hasPath ? (
          <aside className="w-full lg:w-64 lg:shrink-0 bg-white border rounded p-4 mb-8 lg:mb-0">
            <h3 className="text-sm font-semibold mb-3">Generated Path (Goals)</h3>
            <ul className="space-y-2 text-sm">
              {pathNodes?.map((node, idx) => {
                const displayTitle = node.title || node.concept || node.id || node.node_id || `Node ${idx + 1}`;
                const estimate = node.estimated_minutes ?? node.time_estimate;
                const key = node.node_id || node.id || `${displayTitle}-${idx}`;
                return (
                  <li key={key} className="border rounded px-2 py-1">
                    <div className="font-medium">{displayTitle}</div>
                    {typeof estimate === "number" && (
                      <div className="text-xs text-gray-500">Est: {estimate} min</div>
                    )}
                  </li>
                );
              })}
            </ul>
          </aside>
        ) : null}

        <main className={`flex-1 ${hasPath ? "" : "max-w-4xl mx-auto"}`}>
          <div className="mb-6">
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

        <section className="mt-12 border rounded-lg bg-white/80 shadow-sm p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-medium">Coach Chat</h2>
              <p className="text-sm text-gray-600 mt-1">
                Ask follow-up questions and refine goals. Context stays in this session.
              </p>
            </div>
            <div className="text-xs text-gray-500 text-right">
              <div>Model: {chatModel}</div>
              {chatFallback && <div className="text-amber-600">Fallback heuristics used</div>}
            </div>
          </div>

          <div
            ref={chatScrollRef}
            className="mt-4 h-64 border rounded bg-gray-50 overflow-y-auto p-3 space-y-3"
          >
            {chatMessages.length === 0 ? (
              <p className="text-sm text-gray-500">
                No conversation yet. Example: Toi muon on SQL truoc khi hoc MongoDB.
              </p>
            ) : (
              chatMessages.map((msg, idx) => {
                const isUser = msg.role === "user";
                return (
                  <div
                    key={`${msg.timestamp || idx}-${msg.role}-${idx}`}
                    className={`rounded px-3 py-2 text-sm ${
                      isUser ? "bg-blue-100 text-blue-900" : "bg-white text-gray-800"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{isUser ? "You" : "Coach"}</span>
                      {msg.timestamp && (
                        <span className="text-xs text-gray-500">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                      )}
                    </div>
                    <p className="mt-1 whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  </div>
                );
              })
            )}
          </div>

          {chatError && <div className="mt-3 text-sm text-red-600">{chatError}</div>}
          {chatSummary && (
            <div className="mt-3 text-xs text-gray-600">
              <span className="font-semibold">KG summary:</span> {chatSummary}
            </div>
          )}

          {chatInsights.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-semibold text-gray-700">Supporting concepts</h3>
              <ul className="mt-2 text-sm text-gray-700 space-y-1">
                {chatInsights.map((node, idx) => {
                  const title = node.concept || node.title || node.node_id || `Node ${idx + 1}`;
                  const extras: string[] = [];
                  if (node.objective) extras.push(node.objective);
                  if (node.context) extras.push(`context: ${node.context}`);
                  if (typeof node.mastery === "number") extras.push(`mastery ${node.mastery.toFixed(2)}`);
                  if (typeof node.deficiency === "number") extras.push(`gap ${node.deficiency.toFixed(2)}`);
                  return (
                    <li key={node.node_id || `${title}-${idx}`} className="flex flex-col">
                      <span className="font-medium">{title}</span>
                      {extras.length > 0 && (
                        <span className="text-xs text-gray-500">{extras.join(" · ")}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          <form onSubmit={handleChatSubmit} className="mt-6 space-y-3">
            <label className="text-sm font-medium" htmlFor="chatMessage">
              Message
            </label>
            <textarea
              id="chatMessage"
              className="w-full border rounded px-3 py-2 h-24"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="E.g.: Toi muon on SQL truoc khi hoc MongoDB."
            />
            <div className="flex items-center gap-3">
              <button
                type="submit"
                className="px-4 py-2 bg-green-600 text-white rounded disabled:opacity-60"
                disabled={chatLoading}
              >
                {chatLoading ? "Sending…" : "Send"}
              </button>
              <button
                type="button"
                className="text-sm text-gray-600 underline"
                onClick={handleChatReset}
              >
                Reset session
              </button>
              {chatSessionId && (
                <span className="text-xs text-gray-400">Session: {chatSessionId.slice(0, 8)}…</span>
              )}
            </div>
          </form>
        </section>
        </main>
      </div>
    </div>
  );
}
